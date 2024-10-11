use std::mem::ManuallyDrop;

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

pub(crate) fn is_msm_context_initialized() -> bool {
    unsafe { _MSM_BASES.is_some() && is_msm_result_mempool_initialized() }
}

pub(crate) fn drop_msm_context() {
    unsafe {
        let _ = _MSM_BASES.take();
        if let Some(msm_bases_pool) = _MSM_BASES_MEMPOOL.take() {
            let result = gpu_ffi::bc_mem_pool_destroy(msm_bases_pool);
            if result != 0 {
                panic!("Couldn't destroy the bases mempool");
            }
        }

        let result = gpu_ffi::msm_tear_down();
        if result != 0 {
            panic!("Couldn't tear down MSM");
        }
        destroy_msm_result_mempool();
    }
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
    DVec::<u8, PoolAllocator>::allocate_on(num_tmp_bytes, _tmp_mempool(), stream);
}

pub(crate) fn is_tmp_mempool_initialized() -> bool {
    unsafe { _TMP_MEMPOOL.is_some() }
}

pub(crate) fn _tmp_mempool() -> bc_mem_pool {
    unsafe { _TMP_MEMPOOL.expect("tmp mempool intialized") }
}

pub(crate) fn destroy_tmp_mempool() {
    unsafe {
        let pool = _TMP_MEMPOOL.take().unwrap();
        let result = gpu_ffi::bc_mem_pool_destroy(pool);
        if result != 0 {
            panic!("Couldn't destry tmp mempool");
        }
    }
}

pub(crate) static mut _MEMPOOL: Option<bc_mem_pool> = None;
pub(crate) static mut _MSM_BASES: Option<MSMBases> = None;

// MSM bases are stored on a static buffer that is neither
// default static allocator nor pool allocator.
// Drop implementation releases its buffer.
pub struct StaticBasesStorage(ManuallyDrop<DVec<CompactG1Affine, PoolAllocator>>);

impl StaticBasesStorage {
    pub fn allocate(num_bases: usize) -> CudaResult<Self> {
        let num_bytes = num_bases * std::mem::size_of::<CompactG1Affine>();
        let bases_ptr = allocate(num_bytes)?;

        // TODO pool allocator is kind of marker allocator here
        let d_bases = unsafe {
            DVec::<CompactG1Affine, PoolAllocator>::from_raw_parts_in(
                bases_ptr.cast(),
                num_bases,
                PoolAllocator,
                None,
                None,
            )
        };

        Ok(Self(ManuallyDrop::new(d_bases)))
    }
}

impl Drop for StaticBasesStorage {
    fn drop(&mut self) {
        let ptr = self.0.as_mut_ptr();
        dealloc(ptr.cast()).expect("dellocate bases static allocation");
    }
}

pub enum MSMBases {
    Static(StaticBasesStorage),
    Pool(DVec<CompactG1Affine, PoolAllocator>),
}

impl MSMBases {
    pub fn as_ptr(&self) -> *const CompactG1Affine {
        match self {
            MSMBases::Static(storage) => storage.0.as_ptr(),
            MSMBases::Pool(storage) => storage.as_ptr(),
        }
    }
}

impl std::ops::Deref for MSMBases {
    type Target = DSlice<CompactG1Affine>;

    fn deref(&self) -> &Self::Target {
        match self {
            MSMBases::Static(ref storage) => &storage.0,
            MSMBases::Pool(ref storage) => storage,
        }
    }
}

pub struct DeviceContext<const N: usize>;

const POWERS_OF_OMEGA_COARSE_LOG_COUNT: u32 = 25;
const POWERS_OF_COSET_OMEGA_COARSE_LOG_COUNT: u32 = 14;

pub type DeviceContextWithSingleDevice = DeviceContext<1>;

impl<const N: usize> DeviceContext<N> {
    pub fn init(domain_size: usize) -> CudaResult<Self> {
        let context = Self::init_no_msm(domain_size)?;
        Self::init_msm_on_static_memory(domain_size)?;
        // Self::init_msm_on_pool(domain_size)?;

        Ok(context)
    }

    pub fn init_no_msm(domain_size: usize) -> CudaResult<Self> {
        init_allocations(domain_size);
        unsafe {
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
        }

        Ok(DeviceContext)
    }

    fn init_msm_on_static_memory(domain_size: usize) -> CudaResult<()> {
        Self::inner_init_msm(domain_size, None, None)?;
        Ok(())
    }

    // In reality we keep bases on a statically allocated buffer.
    unsafe fn init_msm_on_pool(domain_size: usize) -> CudaResult<()> {
        let pool = _msm_bases_mempool();
        let stream = bc_stream::new().unwrap();
        Self::inner_init_msm(domain_size, Some(pool), Some(stream))?;
        stream.sync().unwrap();
        Ok(())
    }

    fn inner_init_msm(
        domain_size: usize,
        pool: Option<bc_mem_pool>,
        stream: Option<bc_stream>,
    ) -> CudaResult<()> {
        assert!(
            is_msm_context_initialized() == false,
            "MSM context is already initialized"
        );
        init_msm_result_mempool();
        // MSM impl requires bases to be located in a buffer that is
        // multiple of the domain_size
        let crs = init_compact_crs(&bellman::worker::Worker::new(), domain_size);
        let num_bases = MAX_COMBINED_DEGREE_FACTOR * domain_size;
        assert!(crs.g1_bases.len() >= num_bases);
        let bases = match (pool, stream) {
            (Some(pool), Some(stream)) => {
                println!("Transferring bases to the pool");
                let mut bases = DVec::allocate_zeroed_on(num_bases, pool, stream);
                mem::h2d_on(&crs.g1_bases, &mut bases, stream)?;
                stream.sync().unwrap();
                MSMBases::Pool(bases)
            }
            (None, None) => {
                println!("Transferring bases to the static memory of device");
                let mut bases = StaticBasesStorage::allocate(num_bases)?;
                mem::memcopy_from_host(&mut bases.0, &crs.g1_bases[..num_bases])?;
                MSMBases::Static(bases)
            }
            _ => unreachable!(),
        };
        unsafe {
            _MSM_BASES = Some(std::mem::transmute(bases));

            println!("Configuring device for MSM");
            if gpu_ffi::msm_set_up() != 0 {
                return Err(CudaError::SetupError("MSM configuration error".to_string()));
            }
        }

        Ok(())
    }
}

pub fn init_allocations(domain_size: usize) {
    init_static_alloc(domain_size);
    init_small_scalar_mempool();
    init_tmp_mempool();
}

pub fn free_allocations() {
    free_static_alloc();
    destroy_small_scalar_mempool();
    destroy_tmp_mempool();
}

impl<const N: usize> Drop for DeviceContext<N> {
    fn drop(&mut self) {
        unsafe {
            if is_msm_context_initialized() {
                drop_msm_context();
            }
            free_allocations();

            let result = gpu_ffi::pn_tear_down();
            if result != 0 {
                panic!("Couldn't tear down the permutation precomputations");
            }
            let result = gpu_ffi::ntt_tear_down();
            if result != 0 {
                panic!("Couldn't tear down the permutation precomputations");
            }
            let result = gpu_ffi::ff_tear_down();
            if result != 0 {
                panic!("Couldn't tear down the permutation precomputations");
            }
        }
    }
}

pub fn is_context_initialized() -> bool {
    is_msm_context_initialized()
        && is_small_scalar_mempool_initialized()
        && is_tmp_mempool_initialized()
}

pub(crate) fn _bases() -> &'static DSlice<CompactG1Affine> {
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
    } else if let Ok(crs_file_path) = std::env::var("COMPACT_CRS_FILE") {
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
