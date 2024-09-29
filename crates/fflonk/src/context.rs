use super::*;
use bellman::compact_bn256::Bn256 as CompactBn256;
use gpu_ffi::bc_mem_pool;

static mut _MSM_BASES_MEMPOOL: Option<bc_mem_pool> = None;
static mut _TMP_MEMPOOL: Option<bc_mem_pool> = None;

pub(crate) fn init_msm_bases_mempool() {
    assert!(is_msm_bases_mempool_initialized() == false);
    unsafe {
        _MSM_BASES_MEMPOOL = Some(bc_mem_pool::new(DEFAULT_DEVICE_ID).unwrap());
    }
}

pub(crate) fn is_msm_bases_mempool_initialized() -> bool {
    unsafe { _MSM_BASES_MEMPOOL.is_some() }
}

pub(crate) fn _msm_bases_mempool() -> bc_mem_pool {
    unsafe { _MSM_BASES_MEMPOOL.expect("msm bases mempool intialized") }
}

// Temporary buffer specifically for the grand product, batch inv and poly eval
pub(crate) const POLY_EVAL_CHUNK_SIZE: usize = 1 << 28; // 142609151bytes 0,14GB
pub(crate) const _GRAND_PROD_CHUNK_SIZE: usize = 1 << 25; // 1091570431bytes 1,09GB
pub(crate) const BATCH_INV_CHUNK_SIZE: usize = 1 << 25; // 1073741824bytes 1,07GB
pub(crate) const MSM_CHUNK_SIZE: usize = 1 << 23; // 1073741824bytes 1,07GB

pub(crate) fn init_tmp_mempool() {
    assert!(is_tmp_mempool_initialized() == false);
    unsafe {
        _TMP_MEMPOOL = Some(bc_mem_pool::new(DEFAULT_DEVICE_ID).unwrap());
    }
    let num_tmp_bytes = 1_100_000_000;
    let stream = bc_stream::new().unwrap();
    let warmup_buf: DVec<u8> = DVec::allocate_on(num_tmp_bytes, _tmp_mempool(), stream);
    drop(warmup_buf);
}

pub(crate) fn is_tmp_mempool_initialized() -> bool {
    unsafe { _TMP_MEMPOOL.is_some() }
}

pub(crate) fn _tmp_mempool() -> bc_mem_pool {
    unsafe { _TMP_MEMPOOL.expect("tmp mempool intialized") }
}

pub(crate) static mut _MEMPOOL: Option<bc_mem_pool> = None;
pub(crate) static mut _MSM_BASES: Option<DVec<CompactG1Affine>> = None;

pub struct DeviceContext<const N: usize>;

const POWERS_OF_OMEGA_COARSE_LOG_COUNT: u32 = 25;
const POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT: u32 = 14;

pub type DeviceContextWithSingleDevice = DeviceContext<1>;

impl<const N: usize> DeviceContext<N> {
    pub unsafe fn init(domain_size: usize) -> CudaResult<Self> {
        Self::init_msm_on_pool(domain_size)?;

        Self::init_no_msm()
    }

    pub unsafe fn init_no_msm() -> CudaResult<Self> {
        init_small_scalar_mempool();
        init_tmp_mempool();
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

        Ok(DeviceContext)
    }

    unsafe fn init_msm_on_static_memory(domain_size: usize) -> CudaResult<()> {
        assert!(_MSM_BASES.is_none(), "MSM context is already initialized");
        Self::inner_init_msm(domain_size, None, None)?;
        Ok(())
    }

    // In reality we keep bases on a statically allocated buffer.
    unsafe fn init_msm_on_pool(domain_size: usize) -> CudaResult<()> {
        init_msm_bases_mempool();
        let pool = _msm_bases_mempool();
        let stream = bc_stream::new().unwrap();
        Self::inner_init_msm(domain_size, Some(pool), Some(stream))?;
        stream.sync().unwrap();
        Ok(())
    }

    unsafe fn inner_init_msm(
        domain_size: usize,
        pool: Option<bc_mem_pool>,
        stream: Option<bc_stream>,
    ) -> CudaResult<()> {
        init_msm_result_mempool();
        assert!(_MSM_BASES.is_none(), "MSM context is already initialized");
        // MSM impl requires bases to be located in a buffer that is
        // multiple of the domain_size
        let crs = init_compact_crs(&bellman::worker::Worker::new(), domain_size);
        use bellman::CurveAffine;
        let num_bases = MAX_COMBINED_DEGREE_FACTOR * domain_size;
        let d_bases = match (pool, stream) {
            (Some(pool), Some(stream)) => {
                println!("Transferring bases to the pool");
                let mut d_bases = DVec::allocate_zeroed_on(num_bases, pool, stream);
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
            (None, None) => {
                println!("Transferring bases to the static memory of device");
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
            _ => unreachable!(),
        };

        _MSM_BASES = Some(std::mem::transmute(d_bases));

        println!("Configuring device for MSM");
        if gpu_ffi::msm_set_up() != 0 {
            return Err(CudaError::SetupError("MSM configuration error".to_string()));
        }

        Ok(())
    }
}

impl<const N: usize> Drop for DeviceContext<N> {
    fn drop(&mut self) {
        println!("Dropping device context");
        unsafe {
            if let Some(bases) = _MSM_BASES.take() {
                let _ = bases;

                let result = gpu_ffi::msm_tear_down();
                if result != 0 {
                    println!("Couldn't tear down MSM");
                }
                let msm_bases_pool = _MSM_BASES_MEMPOOL.take().unwrap();
                let result = gpu_ffi::bc_mem_pool_destroy(msm_bases_pool);
                if result != 0 {
                    println!("Couldn't destroy the mempool");
                }
            }

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
        }
    }
}

pub fn is_context_initialized() -> bool {
    is_msm_bases_mempool_initialized()
        && is_msm_result_mempool_initialized()
        && is_small_scalar_mempool_initialized()
}

pub(crate) fn _bases() -> &'static DVec<CompactG1Affine> {
    unsafe { _MSM_BASES.as_ref().expect("MSM bases on the device ") }
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
