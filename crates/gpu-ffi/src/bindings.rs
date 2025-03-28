/* automatically generated by rust-bindgen 0.70.1 */

pub const __bool_true_false_are_defined: u32 = 1;
pub const true_: u32 = 1;
pub const false_: u32 = 0;
pub type size_t = ::std::os::raw::c_ulong;
pub type wchar_t = ::std::os::raw::c_int;
#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Copy, Clone)]
pub struct max_align_t {
    pub __clang_max_align_nonce1: ::std::os::raw::c_longlong,
    pub __bindgen_padding_0: u64,
    pub __clang_max_align_nonce2: u128,
}
pub const bc_error_bc_success: bc_error = 0;
pub const bc_error_bc_error_invalid_value: bc_error = 1;
pub const bc_error_bc_error_memory_allocation: bc_error = 2;
pub const bc_error_bc_error_not_ready: bc_error = 600;
pub type bc_error = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct bc_stream {
    pub handle: *mut ::std::os::raw::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct bc_event {
    pub handle: *mut ::std::os::raw::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct bc_mem_pool {
    pub handle: *mut ::std::os::raw::c_void,
}
pub type bc_host_fn =
    ::std::option::Option<unsafe extern "C" fn(user_data: *mut ::std::os::raw::c_void)>;
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_get_device_count(count: *mut ::std::os::raw::c_int) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_get_device(device_id: *mut ::std::os::raw::c_int) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_set_device(device_id: ::std::os::raw::c_int) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_stream_create(stream: *mut bc_stream, blocking_sync: bool) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_stream_wait_event(stream: bc_stream, event: bc_event) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_stream_synchronize(stream: bc_stream) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_stream_query(stream: bc_stream) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_stream_destroy(stream: bc_stream) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_launch_host_fn(
        stream: bc_stream,
        fn_: bc_host_fn,
        user_data: *mut ::std::os::raw::c_void,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_event_create(
        event: *mut bc_event,
        blocking_sync: bool,
        disable_timing: bool,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_event_record(event: bc_event, stream: bc_stream) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_event_synchronize(event: bc_event) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_event_query(event: bc_event) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_event_destroy(event: bc_event) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_event_elapsed_time(ms: *mut f32, start: bc_event, end: bc_event) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_mem_get_info(free: *mut size_t, total: *mut size_t) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_malloc(ptr: *mut *mut ::std::os::raw::c_void, size: size_t) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_malloc_host(ptr: *mut *mut ::std::os::raw::c_void, size: size_t) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_free(ptr: *mut ::std::os::raw::c_void) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_free_host(ptr: *mut ::std::os::raw::c_void) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_host_register(ptr: *mut ::std::os::raw::c_void, size: size_t) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_host_unregister(ptr: *mut ::std::os::raw::c_void) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_device_disable_peer_access(device_id: ::std::os::raw::c_int) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_device_enable_peer_access(device_id: ::std::os::raw::c_int) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_memcpy(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: size_t,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_memcpy_async(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: size_t,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_memset(
        ptr: *mut ::std::os::raw::c_void,
        value: ::std::os::raw::c_int,
        count: size_t,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_memset_async(
        ptr: *mut ::std::os::raw::c_void,
        value: ::std::os::raw::c_int,
        count: size_t,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_mem_pool_create(pool: *mut bc_mem_pool, device_id: ::std::os::raw::c_int)
        -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_mem_pool_destroy(pool: bc_mem_pool) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_mem_pool_disable_peer_access(
        pool: bc_mem_pool,
        device_id: ::std::os::raw::c_int,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_mem_pool_enable_peer_access(
        pool: bc_mem_pool,
        device_id: ::std::os::raw::c_int,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_malloc_from_pool_async(
        ptr: *mut *mut ::std::os::raw::c_void,
        size: size_t,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn bc_free_async(ptr: *mut ::std::os::raw::c_void, stream: bc_stream) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_set_up(
        powers_of_w_coarse_log_count: ::std::os::raw::c_uint,
        powers_of_g_coarse_log_count: ::std::os::raw::c_uint,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_set_value(
        target: *mut ::std::os::raw::c_void,
        value: *const ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_set_value_zero(
        target: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_set_value_one(
        target: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_ax(
        a: *const ::std::os::raw::c_void,
        x: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_a_plus_x(
        a: *const ::std::os::raw::c_void,
        x: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_x_plus_y(
        x: *const ::std::os::raw::c_void,
        y: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_ax_plus_y(
        a: *const ::std::os::raw::c_void,
        x: *const ::std::os::raw::c_void,
        y: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_x_minus_y(
        x: *const ::std::os::raw::c_void,
        y: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_ax_minus_y(
        a: *const ::std::os::raw::c_void,
        x: *const ::std::os::raw::c_void,
        y: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_x_minus_ay(
        a: *const ::std::os::raw::c_void,
        x: *const ::std::os::raw::c_void,
        y: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_x_mul_y(
        x: *const ::std::os::raw::c_void,
        y: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ff_grand_product_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub inputs: *mut ::std::os::raw::c_void,
    pub outputs: *mut ::std::os::raw::c_void,
    pub count: ::std::os::raw::c_uint,
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_grand_product(configuration: ff_grand_product_configuration) -> bc_error;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ff_multiply_by_powers_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub inputs: *mut ::std::os::raw::c_void,
    pub base: *mut ::std::os::raw::c_void,
    pub outputs: *mut ::std::os::raw::c_void,
    pub count: ::std::os::raw::c_uint,
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_multiply_by_powers(configuration: ff_multiply_by_powers_configuration) -> bc_error;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ff_inverse_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub inputs: *mut ::std::os::raw::c_void,
    pub outputs: *mut ::std::os::raw::c_void,
    pub count: ::std::os::raw::c_uint,
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_inverse(configuration: ff_inverse_configuration) -> bc_error;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ff_poly_evaluate_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub values: *mut ::std::os::raw::c_void,
    pub point: *mut ::std::os::raw::c_void,
    pub result: *mut ::std::os::raw::c_void,
    pub count: ::std::os::raw::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ff_sort_u32_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub values: *mut ::std::os::raw::c_void,
    pub sorted_values: *mut ::std::os::raw::c_void,
    pub count: ::std::os::raw::c_uint,
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_poly_evaluate(configuration: ff_poly_evaluate_configuration) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_get_powers_of_w(
        target: *mut ::std::os::raw::c_void,
        log_degree: ::std::os::raw::c_uint,
        offset: ::std::os::raw::c_uint,
        count: ::std::os::raw::c_uint,
        inverse: bool,
        bit_reversed: bool,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_get_powers_of_g(
        target: *mut ::std::os::raw::c_void,
        log_degree: ::std::os::raw::c_uint,
        offset: ::std::os::raw::c_uint,
        count: ::std::os::raw::c_uint,
        inverse: bool,
        bit_reversed: bool,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_omega_shift(
        values: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        log_degree: ::std::os::raw::c_uint,
        shift: ::std::os::raw::c_uint,
        offset: ::std::os::raw::c_uint,
        count: ::std::os::raw::c_uint,
        inverse: bool,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_bit_reverse(
        values: *const ::std::os::raw::c_void,
        result: *mut ::std::os::raw::c_void,
        log_count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_bit_reverse_multigpu(
        values: *mut *const ::std::os::raw::c_void,
        results: *mut *mut ::std::os::raw::c_void,
        log_count: ::std::os::raw::c_uint,
        streams: *const bc_stream,
        device_ids: *const ::std::os::raw::c_int,
        log_devices_count: ::std::os::raw::c_uint,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_select(
        source: *const ::std::os::raw::c_void,
        destination: *mut ::std::os::raw::c_void,
        indexes: *const ::std::os::raw::c_uint,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_sort_u32(configuration: ff_sort_u32_configuration) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ff_tear_down() -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn pn_set_up() -> bc_error;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct generate_permutation_polynomials_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub indexes: *mut ::std::os::raw::c_uint,
    pub scalars: *mut ::std::os::raw::c_void,
    pub target: *mut ::std::os::raw::c_void,
    pub columns_count: ::std::os::raw::c_uint,
    pub log_rows_count: ::std::os::raw::c_uint,
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn pn_generate_permutation_polynomials(
        configuration: generate_permutation_polynomials_configuration,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn pn_set_values_from_packed_bits(
        values: *mut ::std::os::raw::c_void,
        packet_bits: *const ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn pn_distribute_values(
        src: *const ::std::os::raw::c_void,
        dst: *mut ::std::os::raw::c_void,
        count: ::std::os::raw::c_uint,
        stride: ::std::os::raw::c_uint,
        stream: bc_stream,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn pn_tear_down() -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn msm_set_up() -> bc_error;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct msm_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub bases: *mut ::std::os::raw::c_void,
    pub scalars: *mut ::std::os::raw::c_void,
    pub results: *mut ::std::os::raw::c_void,
    pub log_scalars_count: ::std::os::raw::c_uint,
    pub h2d_copy_finished: bc_event,
    pub h2d_copy_finished_callback: bc_host_fn,
    pub h2d_copy_finished_callback_data: *mut ::std::os::raw::c_void,
    pub d2h_copy_finished: bc_event,
    pub d2h_copy_finished_callback: bc_host_fn,
    pub d2h_copy_finished_callback_data: *mut ::std::os::raw::c_void,
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn msm_execute_async(configuration: msm_configuration) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn msm_tear_down() -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ntt_set_up() -> bc_error;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ntt_configuration {
    pub mem_pool: bc_mem_pool,
    pub stream: bc_stream,
    pub inputs: *mut ::std::os::raw::c_void,
    pub outputs: *mut ::std::os::raw::c_void,
    pub log_values_count: ::std::os::raw::c_uint,
    pub bit_reversed_inputs: bool,
    pub inverse: bool,
    pub can_overwrite_inputs: bool,
    pub log_extension_degree: ::std::os::raw::c_uint,
    pub coset_index: ::std::os::raw::c_uint,
    pub h2d_copy_finished: bc_event,
    pub h2d_copy_finished_callback: bc_host_fn,
    pub h2d_copy_finished_callback_data: *mut ::std::os::raw::c_void,
    pub d2h_copy_finished: bc_event,
    pub d2h_copy_finished_callback: bc_host_fn,
    pub d2h_copy_finished_callback_data: *mut ::std::os::raw::c_void,
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ntt_execute_async(configuration: ntt_configuration) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ntt_execute_async_multigpu(
        configurations: *const ntt_configuration,
        dev_ids: *const ::std::os::raw::c_int,
        log_n_devs: ::std::os::raw::c_uint,
    ) -> bc_error;
}
era_cudart_sys::cuda_fn_and_stub! {
    pub fn ntt_tear_down() -> bc_error;
}
