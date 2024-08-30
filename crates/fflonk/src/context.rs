use super::*;
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

    pub unsafe fn init<E: Engine>(bases: &[E::G1Affine], domain_size: usize) -> CudaResult<Self> {
        let context = Self::init_minimal()?;

        Self::init_msm_on_static_memory::<E>(bases, domain_size)?;

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

    pub unsafe fn init_msm_only<E: Engine>(
        bases: &[E::G1Affine],
        domain_size: usize,
    ) -> CudaResult<Self> {
        let context = Self::init_minimal()?;
        Self::init_msm_on_static_memory::<E>(bases, domain_size)?;
        _h2d_stream().sync().unwrap();

        Ok(context)
    }

    unsafe fn init_msm_on_static_memory<E: Engine>(
        bases: &[E::G1Affine],
        domain_size: usize,
    ) -> CudaResult<()> {
        assert!(_BASES.is_none(), "MSM context is already initialized");
        Self::init_msm::<E>(bases, domain_size, None)?;

        Ok(())
    }
    // In reality we keep bases on a statically allocated buffer.
    unsafe fn init_msm_on_async<E: Engine>(
        bases: &[E::G1Affine],
        domain_size: usize,
    ) -> CudaResult<()> {
        Self::init_msm::<E>(bases, domain_size, Some(_h2d_stream()))?;

        Ok(())
    }

    unsafe fn init_msm<E: Engine>(
        bases: &[E::G1Affine],
        domain_size: usize,
        stream: Option<bc_stream>,
    ) -> CudaResult<()> {
        assert!(
            std::mem::size_of_val(&bases[0]) == 64 || std::mem::size_of_val(&bases[0]) == 72,
            "Provided bases aren't valid"
        );
        assert_eq!(bases.is_empty(), false);

        println!("Transferring bases to the static memory of device");
        let padded_size = bases.len().next_multiple_of(domain_size);
        // MSM impl requires bases to be located in a buffer that is
        // multiple of the domain_size
        let d_bases = match stream {
            Some(stream) => {
                let mut d_bases = DVec::with_capacity_zeroed_on(padded_size, stream);
                mem::h2d_on_stream(&bases, &mut d_bases[..bases.len()], stream)?;
                stream.sync().unwrap();
                d_bases
            }
            None => {
                let mut d_bases = DVec::with_capacity_zeroed(padded_size);
                mem::memcopy_from_host(&mut d_bases, bases)?;
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
    unsafe { _MEMPOOL.as_ref().expect("mempool").clone() }
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
