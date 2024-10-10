use super::*;
use boojum_cuda::context::Context;
use era_cudart::device::{device_get_attribute, get_device};
use era_cudart::event::{CudaEvent, CudaEventCreateFlags};
use era_cudart::memory::{CudaHostAllocFlags, HostAllocation};
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStreamCreateFlags;
use era_cudart_sys::CudaDeviceAttr;
use std::any::{Any, TypeId};
use std::collections::HashMap;

pub(crate) const NUM_AUX_STREAMS_AND_EVENTS: usize = 4;
pub(crate) const AUX_H2D_BUFFER_SIZE: usize = 1 << 24; // 16 MB

#[allow(dead_code)]
struct ProverContextSingleton {
    cuda_context: CudaContext,
    exec_stream: Stream,
    h2d_stream: Stream,
    d2h_stream: Stream,
    device_allocator: StaticDeviceAllocator,
    small_device_allocator: SmallStaticDeviceAllocator,
    host_allocator: StaticHostAllocator,
    small_host_allocator: SmallStaticHostAllocator,
    setup_cache: Option<(TypeId, Box<dyn Any>)>,
    strategy_cache: HashMap<u64, CacheStrategy>,
    l2_cache_size: usize,
    l2_persist_max: usize,
    compute_capability: (u32, u32),
    aux_streams: [CudaStream; NUM_AUX_STREAMS_AND_EVENTS],
    aux_events: [CudaEvent; NUM_AUX_STREAMS_AND_EVENTS],
    aux_h2d_buffer: HostAllocation<u8>,
}

static mut CONTEXT: Option<ProverContextSingleton> = None;

pub struct ProverContext;

pub const ZKSYNC_DEFAULT_TRACE_LOG_LENGTH: u32 = 20;

#[derive(Copy, Clone, Debug)]
pub struct ProverContextConfig {
    // minimum and maximum device allocations are in bytes
    minimum_device_allocation: Option<usize>,
    maximum_device_allocation: Option<usize>,
    smallest_supported_domain_size: usize,
    powers_of_w_coarse_log_count: u32,
    powers_of_g_coarse_log_count: u32,
}

impl Default for ProverContextConfig {
    fn default() -> Self {
        Self {
            minimum_device_allocation: None,
            maximum_device_allocation: None,
            smallest_supported_domain_size: 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH,
            powers_of_w_coarse_log_count: 15,
            powers_of_g_coarse_log_count: 15,
        }
    }
}

impl ProverContextConfig {
    pub fn with_minimum_device_allocation(mut self, minimum_device_allocation: usize) -> Self {
        self.minimum_device_allocation = Some(minimum_device_allocation);
        self
    }

    pub fn with_maximum_device_allocation(mut self, maximum_device_allocation: usize) -> Self {
        self.maximum_device_allocation = Some(maximum_device_allocation);
        self
    }

    pub fn with_smallest_supported_domain_size(
        mut self,
        smallest_supported_domain_size: usize,
    ) -> Self {
        assert!(smallest_supported_domain_size.is_power_of_two());
        self.smallest_supported_domain_size = smallest_supported_domain_size;
        self
    }

    pub fn with_powers_of_w_coarse_log_count(mut self, powers_of_w_coarse_log_count: u32) -> Self {
        self.powers_of_w_coarse_log_count = powers_of_w_coarse_log_count;
        self
    }

    pub fn with_powers_of_g_coarse_log_count(mut self, powers_of_g_coarse_log_count: u32) -> Self {
        self.powers_of_g_coarse_log_count = powers_of_g_coarse_log_count;
        self
    }
}

impl ProverContext {
    fn create_internal(
        cuda_context: Context,
        small_device_allocator: SmallStaticDeviceAllocator,
        device_allocator: StaticDeviceAllocator,
        small_host_allocator: SmallStaticHostAllocator,
        host_allocator: StaticHostAllocator,
    ) -> CudaResult<Self> {
        unsafe {
            assert!(CONTEXT.is_none());
            let device_id = get_device()?;
            let l2_cache_size =
                device_get_attribute(CudaDeviceAttr::L2CacheSize, device_id)? as usize;
            let l2_persist_max =
                device_get_attribute(CudaDeviceAttr::MaxPersistingL2CacheSize, device_id)? as usize;
            let compute_capability_major =
                device_get_attribute(CudaDeviceAttr::ComputeCapabilityMajor, device_id)? as u32;
            let compute_capability_minor =
                device_get_attribute(CudaDeviceAttr::ComputeCapabilityMinor, device_id)? as u32;
            let compute_capability = (compute_capability_major, compute_capability_minor);
            let aux_streams = (0..NUM_AUX_STREAMS_AND_EVENTS)
                .map(|_| CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING))
                .collect::<CudaResult<Vec<_>>>()?
                .try_into()
                .unwrap();
            let aux_events = (0..NUM_AUX_STREAMS_AND_EVENTS)
                .map(|_| CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING))
                .collect::<CudaResult<Vec<_>>>()?
                .try_into()
                .unwrap();
            let aux_h2d_buffer =
                HostAllocation::alloc(AUX_H2D_BUFFER_SIZE, CudaHostAllocFlags::DEFAULT)?;
            CONTEXT = Some(ProverContextSingleton {
                cuda_context,
                exec_stream: Stream::create()?,
                h2d_stream: Stream::create()?,
                d2h_stream: Stream::create()?,
                device_allocator,
                small_device_allocator,
                host_allocator,
                small_host_allocator,
                setup_cache: None,
                strategy_cache: HashMap::new(),
                l2_cache_size,
                l2_persist_max,
                compute_capability,
                aux_streams,
                aux_events,
                aux_h2d_buffer,
            });
            if l2_persist_max != 0 {
                // 10 sets of powers * 2X safety margin
                set_l2_persistence_carveout(2 * 10 * 8 * (1 << 12))?;
                set_l2_persistence_for_twiddles(get_stream())?;
                for stream in _aux_streams() {
                    set_l2_persistence_for_twiddles(stream)?;
                }
            }
        };
        Ok(Self {})
    }

    pub fn create() -> CudaResult<Self> {
        Self::create_with_config(ProverContextConfig::default())
    }

    pub fn create_with_config(config: ProverContextConfig) -> CudaResult<Self> {
        // size counts in field elements
        let block_size = config.smallest_supported_domain_size;
        let block_size_in_bytes = block_size * size_of::<F>();
        let cuda_ctx = CudaContext::create(12, 12)?;
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let min_num_blocks = if let Some(min) = config.minimum_device_allocation {
            min / block_size_in_bytes
        } else {
            DEFAULT_MIN_NUM_BLOCKS
        };
        let device_alloc = if let Some(max) = config.maximum_device_allocation {
            let max_num_blocks = max / block_size_in_bytes;
            StaticDeviceAllocator::init(min_num_blocks, max_num_blocks, block_size)?
        } else {
            StaticDeviceAllocator::init_all(min_num_blocks, block_size)?
        };
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;
        Self::create_internal(
            cuda_ctx,
            small_device_alloc,
            device_alloc,
            small_host_alloc,
            host_alloc,
        )
    }
}

impl Drop for ProverContext {
    fn drop(&mut self) {
        drop(unsafe { CONTEXT.take() })
    }
}

// Should some of this logic live in boojum-cuda instead?
fn set_l2_persistence_carveout(num_bytes: usize) -> CudaResult<()> {
    use era_cudart::device::device_set_limit;
    use era_cudart_sys::CudaLimit;
    let l2_persist_max = get_context().l2_persist_max;
    let carveout = std::cmp::min(num_bytes, l2_persist_max);
    device_set_limit(CudaLimit::PersistingL2CacheSize, carveout)?;
    Ok(())
}

fn set_l2_persistence(data: &DeviceSlice<F>, stream: &CudaStream) -> CudaResult<()> {
    use era_cudart::execution::CudaLaunchAttribute;
    use era_cudart_sys::CudaAccessPolicyWindow;
    use era_cudart_sys::CudaAccessProperty;
    let num_bytes = 8 * data.len();
    let stream_attribute = CudaLaunchAttribute::AccessPolicyWindow(CudaAccessPolicyWindow {
        base_ptr: data.as_ptr() as *mut std::os::raw::c_void,
        num_bytes,
        hitRatio: 1.0,
        hitProp: CudaAccessProperty::Persisting,
        missProp: CudaAccessProperty::Streaming,
    });
    stream.set_attribute(stream_attribute)?;
    Ok(())
}

fn set_l2_persistence_for_twiddles(stream: &CudaStream) -> CudaResult<()> {
    let ctx = &get_context().cuda_context;
    set_l2_persistence(ctx.powers_of_w_fine.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_w_coarse.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_w_fine_bitrev_for_ntt.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_w_coarse_bitrev_for_ntt.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_w_inv_fine_bitrev_for_ntt.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_w_inv_coarse_bitrev_for_ntt.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_g_f_fine.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_g_f_coarse.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_g_i_fine.as_ref(), stream)?;
    set_l2_persistence(ctx.powers_of_g_i_coarse.as_ref(), stream)?;
    Ok(())
}

fn get_context() -> &'static ProverContextSingleton {
    unsafe { CONTEXT.as_ref().expect("prover context") }
}

fn get_context_mut() -> &'static mut ProverContextSingleton {
    unsafe { CONTEXT.as_mut().expect("prover context") }
}

pub(crate) fn get_stream() -> &'static CudaStream {
    &get_context().exec_stream.inner
}

pub(crate) fn get_h2d_stream() -> &'static CudaStream {
    // &get_context().h2d_stream.inner
    get_stream()
}

pub(crate) fn get_d2h_stream() -> &'static CudaStream {
    // &get_context().d2h_stream.inner
    get_stream()
}

pub fn synchronize_streams() -> CudaResult<()> {
    if_not_dry_run! {
        get_stream().synchronize()?;
        get_h2d_stream().synchronize()?;
        get_d2h_stream().synchronize()
    }
}

// use custom wrapper to work around send + sync requirement of static var
pub struct Stream {
    inner: CudaStream,
}

impl Stream {
    pub fn create() -> CudaResult<Self> {
        Ok(Self {
            inner: CudaStream::create()?,
        })
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

pub(crate) fn _alloc() -> &'static StaticDeviceAllocator {
    &get_context().device_allocator
}

pub(crate) fn _small_alloc() -> &'static SmallStaticDeviceAllocator {
    &get_context().small_device_allocator
}
pub(crate) fn _host_alloc() -> &'static StaticHostAllocator {
    &get_context().host_allocator
}

pub(crate) fn _small_host_alloc() -> &'static SmallStaticHostAllocator {
    &get_context().small_host_allocator
}

pub(crate) fn _setup_cache_get<H: GpuTreeHasher>() -> Option<&'static mut SetupCache<H>> {
    get_context_mut()
        .setup_cache
        .as_mut()
        .filter(|(id, _)| id == &TypeId::of::<H>())
        .map(|(_, box_any)| box_any.downcast_mut::<SetupCache<H>>().unwrap())
}

pub(crate) fn _setup_cache_set<H: GpuTreeHasher>(value: SetupCache<H>) {
    assert!(get_context_mut().setup_cache.is_none());
    get_context_mut().setup_cache = Some((TypeId::of::<H>(), Box::new(value)));
}

pub(crate) fn _setup_cache_reset() {
    get_context_mut().setup_cache = None;
}

pub(crate) fn _strategy_cache_get() -> &'static mut HashMap<u64, CacheStrategy> {
    &mut get_context_mut().strategy_cache
}
pub(crate) fn _strategy_cache_reset() {
    get_context_mut().strategy_cache.clear();
}

pub(crate) fn is_prover_context_initialized() -> bool {
    unsafe { CONTEXT.is_some() }
}

pub(crate) fn _l2_cache_size() -> usize {
    get_context().l2_cache_size
}

pub(crate) fn _compute_capability() -> (u32, u32) {
    get_context().compute_capability
}

pub(crate) fn _aux_streams() -> &'static [CudaStream; NUM_AUX_STREAMS_AND_EVENTS] {
    &get_context().aux_streams
}

pub(crate) fn _aux_events() -> &'static [CudaEvent; NUM_AUX_STREAMS_AND_EVENTS] {
    &get_context().aux_events
}

pub(crate) fn _aux_h2d_buffer() -> &'static mut HostAllocation<u8> {
    &mut get_context_mut().aux_h2d_buffer
}
