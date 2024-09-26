use super::*;
use bellman::compact_bn256::Bn256 as CompactBn256;
use gpu_ffi::bc_mem_pool;

pub(crate) static mut _MEMPOOL: Option<bc_mem_pool> = None;
pub(crate) static mut _BASES: Option<DVec<CompactG1Affine>> = None;
pub(crate) static mut _H2D_STREAM: Option<bc_stream> = None;
pub(crate) static mut _D2H_STREAM: Option<bc_stream> = None;
pub(crate) static mut _D2D_STREAM: Option<bc_stream> = None;

pub struct DeviceContext<const N: usize>;

const POWERS_OF_OMEGA_COARSE_LOG_COUNT: u32 = 25;
const POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT: u32 = 14;

pub type DeviceContextWithSingleDevice = DeviceContext<1>;

impl<const N: usize> DeviceContext<N> {
    pub unsafe fn init_minimal() -> CudaResult<Self> {
        assert_eq!(
            is_context_initialized(),
            false,
            "Context is already initialized"
        );
        let device_id = 0;
        let mempool = bc_mem_pool::new(device_id).unwrap();
        println!("Initializing context variables");
        _MEMPOOL = Some(mempool);
        _H2D_STREAM = Some(bc_stream::new().map_err(|err| CudaError::Error("".to_string()))?);
        _D2H_STREAM = Some(bc_stream::new().map_err(|err| CudaError::Error("".to_string()))?);
        _D2D_STREAM = Some(bc_stream::new().map_err(|err| CudaError::Error("".to_string()))?);

        Ok(Self)
    }

    pub unsafe fn init(domain_size: usize) -> CudaResult<Self> {
        assert_eq!(
            is_context_initialized(),
            false,
            "Context is already initialized"
        );

        let context = Self::init_minimal()?;

        Self::init_msm_on_static_memory(domain_size)?;

        let result = gpu_ffi::ff_set_up(
            POWERS_OF_OMEGA_COARSE_LOG_COUNT,
            POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("FF Setup Error: {}", result)));
        }

        let result = gpu_ffi::ntt_set_up();
        if result != 0 {
            return Err(CudaError::Error(format!("NTT Setup Error: {}", result)));
        }

        let result = gpu_ffi::pn_set_up();
        if result != 0 {
            return Err(CudaError::Error(format!("PN Setup Error: {}", result)));
        }

        Ok(context)
    }

    pub unsafe fn init_no_msm() -> CudaResult<Self> {
        let context = Self::init_minimal()?;

        let result = gpu_ffi::ff_set_up(
            POWERS_OF_OMEGA_COARSE_LOG_COUNT,
            POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("FF Setup Error: {}", result)));
        }

        let result = gpu_ffi::ntt_set_up();
        if result != 0 {
            return Err(CudaError::Error(format!("NTT Setup Error: {}", result)));
        }

        let result = gpu_ffi::pn_set_up();
        if result != 0 {
            return Err(CudaError::Error(format!("PN Setup Error: {}", result)));
        }

        Ok(context)
    }

    pub unsafe fn init_msm_only<E: Engine>(domain_size: usize) -> CudaResult<Self> {
        let context = Self::init_minimal()?;
        Self::init_msm_on_static_memory(domain_size)?;
        _h2d_stream().sync().unwrap();

        Ok(context)
    }

    unsafe fn init_msm_on_static_memory(domain_size: usize) -> CudaResult<()> {
        assert!(_BASES.is_none(), "MSM context is already initialized");
        Self::init_msm(domain_size, None)?;

        Ok(())
    }
    // In reality we keep bases on a statically allocated buffer.
    unsafe fn init_msm_on_async(domain_size: usize) -> CudaResult<()> {
        Self::init_msm(domain_size, Some(_h2d_stream()))?;

        Ok(())
    }

    unsafe fn init_msm(domain_size: usize, stream: Option<bc_stream>) -> CudaResult<()> {
        assert!(_BASES.is_none(), "MSM context is already initialized");
        println!("Transferring bases to the static memory of device");
        // MSM impl requires bases to be located in a buffer that is
        // multiple of the domain_size
        let crs = init_compact_crs(&bellman::worker::Worker::new(), domain_size);
        use bellman::CurveAffine;
        let num_bases = MAX_COMBINED_DEGREE_FACTOR * domain_size;
        let d_bases = match stream {
            Some(stream) => {
                let mut d_bases = DVec::allocate_zeroed_on(num_bases, stream);
                let (actual_bases, remainder) = d_bases.split_at_mut(crs.g1_bases.len());
                mem::h2d_on(&crs.g1_bases, actual_bases, stream)?;
                if !remainder.is_empty() {
                    mem::h2d_on(
                        &vec![CompactG1Affine::zero(); remainder.len()],
                        remainder,
                        stream,
                    )?;
                }

                stream.sync().unwrap();
                d_bases
            }
            None => {
                let mut d_bases = DVec::allocate_zeroed(num_bases);
                let (actual_bases, remainder) = d_bases.split_at_mut(crs.g1_bases.len());
                mem::memcopy_from_host(actual_bases, &crs.g1_bases)?;
                if !remainder.is_empty() {
                    mem::memcopy_from_host(
                        remainder,
                        &vec![CompactG1Affine::zero(); remainder.len()],
                    )?;
                }
                d_bases
            }
        };

        _BASES = Some(std::mem::transmute(d_bases));

        println!("Configuring device for MSM");
        if gpu_ffi::msm_set_up() != 0 {
            return Err(CudaError::SetupError("MSM configuration error".to_string()));
        }

        Ok(())
    }
}

impl<const N: usize> Drop for DeviceContext<N> {
    fn drop(&mut self) {
        unsafe {
            if let Some(bases) = _BASES.take() {
                let _ = bases;

                let result = gpu_ffi::msm_tear_down();
                if result != 0 {
                    println!("Couldn't tear down MSM");
                }
            }

            let _ = _H2D_STREAM.take().unwrap();
            let _ = _D2H_STREAM.take().unwrap();
            let _ = _D2D_STREAM.take().unwrap();

            let result = gpu_ffi::pn_tear_down();
            if result != 0 {
                println!("Couldn't tear down the permutation precomputations");
            }
            let result = gpu_ffi::ntt_tear_down();
            if result != 0 {
                println!("Couldn't tear down the permutation precomputations");
            }
            let result = gpu_ffi::ff_tear_down();
            if result != 0 {
                println!("Couldn't tear down the permutation precomputations");
            }

            let mempool = _MEMPOOL.take().unwrap();
            let result = gpu_ffi::bc_mem_pool_destroy(mempool);
            if result != 0 {
                println!("Couldn't destroy the mempool");
            }
        }
    }
}

pub fn is_context_initialized() -> bool {
    unsafe { _MEMPOOL.is_some() }
}

pub(crate) fn _mem_pool() -> bc_mem_pool {
    unsafe { *_MEMPOOL.as_ref().expect("mempool") }
}

pub(crate) fn _bases() -> &'static DVec<CompactG1Affine> {
    unsafe { _BASES.as_ref().expect("MSM bases on the device ") }
}

pub(crate) fn _h2d_stream() -> bc_stream {
    unsafe { *_H2D_STREAM.as_ref().expect("h2d stream") }
}

pub(crate) fn _d2h_stream() -> bc_stream {
    unsafe { *_D2H_STREAM.as_ref().expect("d2h stream") }
}

pub(crate) fn _d2d_stream() -> bc_stream {
    unsafe { *_D2D_STREAM.as_ref().expect("d2h stream") }
}

use bellman::kate_commitment::{Crs, CrsForMonomialForm};

pub fn init_compact_crs(
    worker: &bellman::worker::Worker,
    domain_size: usize,
) -> Crs<CompactBn256, CrsForMonomialForm> {
    assert!(domain_size <= 1 << fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG);
    let num_points = MAX_COMBINED_DEGREE_FACTOR * domain_size;
    let mon_crs = if let Ok(socket_path) = std::env::var("TEST_SOCK_FILE") {
        read_crs_over_socket(&socket_path).unwrap()
    } else if let Ok(crs_file_path) = std::env::var("CRS_FILE") {
        println!("using crs file at {crs_file_path}");
        let crs_file =
            std::fs::File::open(&crs_file_path).expect(&format!("crs file at {}", crs_file_path));
        let mon_crs = Crs::<CompactBn256, CrsForMonomialForm>::read(crs_file)
            .expect(&format!("read crs file at {}", crs_file_path));
        assert!(num_points <= mon_crs.g1_bases.len());

        mon_crs
    } else {
        Crs::<CompactBn256, CrsForMonomialForm>::non_power_of_two_crs_42(num_points, &worker)
    };

    mon_crs
}

// This is convenient for faster testing
fn read_crs_over_socket(
    socket_path: &str,
) -> std::io::Result<Crs<CompactBn256, CrsForMonomialForm>> {
    let mut socket = std::os::unix::net::UnixStream::connect(socket_path)?;
    let start = std::time::Instant::now();
    std::io::Write::write_all(&mut socket, &[1])?;
    let crs = Crs::<CompactBn256, CrsForMonomialForm>::read(&socket)?;
    println!("Loading CRS takes {} s", start.elapsed().as_secs());

    Ok(crs)
}
