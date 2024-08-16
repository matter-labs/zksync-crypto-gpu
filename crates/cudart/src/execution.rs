// execution control
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html

use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr::null_mut;
use std::sync::{Arc, Weak};

use era_cudart_sys::*;

use crate::result::{CudaResult, CudaResultWrap};
use crate::stream::CudaStream;

#[derive(Debug, Copy, Clone)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

impl From<u32> for Dim3 {
    fn from(value: u32) -> Self {
        Self::new(value, 1, 1)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from(value: (u32, u32)) -> Self {
        Self::new(value.0, value.1, 1)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from(value: (u32, u32, u32)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl From<Dim3> for dim3 {
    fn from(val: Dim3) -> Self {
        Self {
            x: val.x,
            y: val.y,
            z: val.z,
        }
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct CudaLaunchConfig<'a> {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub dynamic_smem_bytes: usize,
    pub stream: Option<&'a CudaStream>,
    pub attributes: &'a [CudaLaunchAttribute],
}

impl<'a> CudaLaunchConfig<'a> {
    pub fn builder() -> CudaLaunchConfigBuilder<'a> {
        CudaLaunchConfigBuilder::new()
    }

    pub fn basic(
        grid_dim: impl Into<Dim3>,
        block_dim: impl Into<Dim3>,
        stream: &'a CudaStream,
    ) -> Self {
        CudaLaunchConfig {
            grid_dim: grid_dim.into(),
            block_dim: block_dim.into(),
            dynamic_smem_bytes: 0,
            stream: Some(stream),
            attributes: &[],
        }
    }
}

#[derive(Default)]
pub struct CudaLaunchConfigBuilder<'a> {
    grid_dim: Dim3,
    block_dim: Dim3,
    dynamic_smem_bytes: usize,
    stream: Option<&'a CudaStream>,
    attributes: &'a [CudaLaunchAttribute],
}

impl<'a> CudaLaunchConfigBuilder<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn grid_dim(mut self, grid_dim: impl Into<Dim3>) -> Self {
        self.grid_dim = grid_dim.into();
        self
    }

    pub fn block_dim(mut self, block_dim: impl Into<Dim3>) -> Self {
        self.block_dim = block_dim.into();
        self
    }

    pub fn dynamic_smem_bytes(mut self, dynamic_smem_bytes: usize) -> Self {
        self.dynamic_smem_bytes = dynamic_smem_bytes;
        self
    }

    pub fn stream(mut self, stream: &'a CudaStream) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn attributes(mut self, attributes: &'a [CudaLaunchAttribute]) -> Self {
        self.attributes = attributes;
        self
    }

    pub fn build(self) -> CudaLaunchConfig<'a> {
        CudaLaunchConfig {
            grid_dim: self.grid_dim,
            block_dim: self.block_dim,
            dynamic_smem_bytes: self.dynamic_smem_bytes,
            stream: self.stream,
            attributes: self.attributes,
        }
    }
}

#[derive(Default)]
pub struct RawKernelArguments<'a> {
    vec: Vec<*mut c_void>,
    phantom: PhantomData<&'a c_void>,
}

impl<'a> RawKernelArguments<'a> {
    pub fn push<T>(&mut self, value: &T) {
        self.vec.push(value as *const T as *mut c_void);
    }

    pub fn as_mut_ptr(&mut self) -> *mut *mut c_void {
        self.vec.as_mut_ptr()
    }
}

#[macro_export]
macro_rules! raw_kernel_arguments {
    ($($x:expr),* $(,)?) => {
        {
            let mut args = $crate::execution::RawKernelArguments::default();
            $(
            args.push($x);
            )*
            args
        }
    };
}

pub trait KernelArguments {
    type Signature;
    fn as_raw(&self) -> RawKernelArguments;
}

#[derive(Debug, Copy, Clone)]
pub enum CudaLaunchAttribute {
    Ignore,
    AccessPolicyWindow(CudaAccessPolicyWindow),
    Cooperative(bool),
    SynchronizationPolicy(CudaSynchronizationPolicy),
    ClusterDimension(dim3),
    ClusterSchedulingPolicyPreference(CudaClusterSchedulingPolicy),
    ProgrammaticStreamSerialization(bool),
    ProgrammaticEvent(cudaLaunchAttributeValue__bindgen_ty_2),
    Priority(i32),
    MemSyncDomainMap(cudaLaunchMemSyncDomainMap),
    MemSyncDomain(CudaLaunchMemSyncDomain),
}

impl CudaLaunchAttribute {
    pub(crate) fn from_id_and_value(
        id: CudaLaunchAttributeID,
        value: CudaLaunchAttributeValue,
    ) -> Self {
        unsafe {
            match id {
                CudaLaunchAttributeID::Ignore => Self::Ignore,
                CudaLaunchAttributeID::AccessPolicyWindow => {
                    Self::AccessPolicyWindow(value.accessPolicyWindow)
                }
                CudaLaunchAttributeID::Cooperative => Self::Cooperative(value.cooperative != 0),
                CudaLaunchAttributeID::SynchronizationPolicy => {
                    Self::SynchronizationPolicy(value.syncPolicy)
                }
                CudaLaunchAttributeID::ClusterDimension => {
                    let d = value.clusterDim;
                    Self::ClusterDimension(dim3 {
                        x: d.x,
                        y: d.y,
                        z: d.z,
                    })
                }
                CudaLaunchAttributeID::ClusterSchedulingPolicyPreference => {
                    Self::ClusterSchedulingPolicyPreference(value.clusterSchedulingPolicyPreference)
                }
                CudaLaunchAttributeID::ProgrammaticStreamSerialization => {
                    Self::ProgrammaticStreamSerialization(
                        value.programmaticStreamSerializationAllowed != 0,
                    )
                }
                CudaLaunchAttributeID::ProgrammaticEvent => {
                    Self::ProgrammaticEvent(value.programmaticEvent)
                }
                CudaLaunchAttributeID::Priority => Self::Priority(value.priority),
                CudaLaunchAttributeID::MemSyncDomainMap => {
                    Self::MemSyncDomainMap(value.memSyncDomainMap)
                }
                CudaLaunchAttributeID::MemSyncDomain => Self::MemSyncDomain(value.memSyncDomain),
                #[allow(unreachable_patterns)]
                _ => unimplemented!("Unsupported CudaLaunchAttributeID"),
            }
        }
    }

    pub(crate) fn into_id_and_value(self) -> (CudaLaunchAttributeID, CudaLaunchAttributeValue) {
        match self {
            CudaLaunchAttribute::Ignore => (
                CudaLaunchAttributeID::Ignore,
                CudaLaunchAttributeValue { pad: [0; 64] },
            ),
            CudaLaunchAttribute::AccessPolicyWindow(access_policy_window) => (
                CudaLaunchAttributeID::AccessPolicyWindow,
                CudaLaunchAttributeValue {
                    accessPolicyWindow: access_policy_window,
                },
            ),
            CudaLaunchAttribute::Cooperative(cooperative) => (
                CudaLaunchAttributeID::Cooperative,
                CudaLaunchAttributeValue {
                    cooperative: cooperative as c_int,
                },
            ),
            CudaLaunchAttribute::SynchronizationPolicy(sync_policy) => (
                CudaLaunchAttributeID::SynchronizationPolicy,
                CudaLaunchAttributeValue {
                    syncPolicy: sync_policy,
                },
            ),
            CudaLaunchAttribute::ClusterDimension(cluster_dim) => (
                CudaLaunchAttributeID::ClusterDimension,
                CudaLaunchAttributeValue {
                    clusterDim: cudaLaunchAttributeValue__bindgen_ty_1 {
                        x: cluster_dim.x,
                        y: cluster_dim.y,
                        z: cluster_dim.z,
                    },
                },
            ),
            CudaLaunchAttribute::ClusterSchedulingPolicyPreference(
                cluster_scheduling_policy_preference,
            ) => (
                CudaLaunchAttributeID::ClusterSchedulingPolicyPreference,
                CudaLaunchAttributeValue {
                    clusterSchedulingPolicyPreference: cluster_scheduling_policy_preference,
                },
            ),
            CudaLaunchAttribute::ProgrammaticStreamSerialization(
                programmatic_stream_serialization_allowed,
            ) => (
                CudaLaunchAttributeID::ProgrammaticStreamSerialization,
                CudaLaunchAttributeValue {
                    programmaticStreamSerializationAllowed:
                        programmatic_stream_serialization_allowed as c_int,
                },
            ),
            CudaLaunchAttribute::ProgrammaticEvent(programmatic_event) => (
                CudaLaunchAttributeID::ProgrammaticEvent,
                CudaLaunchAttributeValue {
                    programmaticEvent: programmatic_event,
                },
            ),
            CudaLaunchAttribute::Priority(priority) => (
                CudaLaunchAttributeID::Priority,
                CudaLaunchAttributeValue { priority },
            ),
            CudaLaunchAttribute::MemSyncDomainMap(mem_sync_domain_map) => (
                CudaLaunchAttributeID::MemSyncDomainMap,
                CudaLaunchAttributeValue {
                    memSyncDomainMap: mem_sync_domain_map,
                },
            ),
            CudaLaunchAttribute::MemSyncDomain(mem_sync_domain) => (
                CudaLaunchAttributeID::MemSyncDomain,
                CudaLaunchAttributeValue {
                    memSyncDomain: mem_sync_domain,
                },
            ),
        }
    }
}

impl From<CudaLaunchAttribute> for cudaLaunchAttribute {
    fn from(val: CudaLaunchAttribute) -> Self {
        let (id, val) = val.into_id_and_value();
        Self {
            id,
            pad: [c_char::default(); 4],
            val,
        }
    }
}

pub trait KernelFunction {
    type Signature;

    fn as_ptr(&self) -> *const c_void;

    fn launch(
        &self,
        config: &CudaLaunchConfig,
        args: &impl KernelArguments<Signature = Self::Signature>,
    ) -> CudaResult<()> {
        let mut attributes = config
            .attributes
            .iter()
            .map(|&attribute| attribute.into())
            .collect::<Vec<_>>();
        let config = cudaLaunchConfig_t {
            gridDim: config.grid_dim.into(),
            blockDim: config.block_dim.into(),
            dynamicSmemBytes: config.dynamic_smem_bytes,
            stream: config.stream.map_or(null_mut(), |s| s.into()),
            attrs: attributes.as_mut_ptr(),
            numAttrs: attributes.len() as c_uint,
        };
        unsafe {
            cudaLaunchKernelExC(
                &config as *const _,
                self.as_ptr(),
                args.as_raw().as_mut_ptr(),
            )
            .wrap()
        }
    }
}

#[macro_export]
macro_rules! cuda_kernel_signature {
    ($vis:vis $name:ident$(<$($gen:tt),+>)?, $($arg_ident:ident:$arg_ty:ty),*$(,)?) => {
        $vis type $name$(<$($gen),*>)? = unsafe extern "C" fn($($arg_ident:$arg_ty),*);
    };
}

#[macro_export]
macro_rules! cuda_kernel_arguments {
    ($vis:vis $name:ident$(<$($gen:tt$(:$gen_tr:tt)?),+>)?, $signature_name:ident, $($arg_ident:ident:$arg_ty:ty),*$(,)?) => {
        $vis struct $name$(<$($gen$(:$gen_tr)?),*>)? {$($arg_ident:$arg_ty,)*}
        impl$(<$($gen$(:$gen_tr)?),*>)? $name$(<$($gen),*>)? {
            #[allow(clippy::too_many_arguments)]
            pub fn new($($arg_ident:$arg_ty,)*) -> Self { Self {$($arg_ident,)*} }
        }
        impl$(<$($gen$(:$gen_tr)?),*>)? $crate::execution::KernelArguments for $name$(<$($gen),*>)? {
            type Signature = $signature_name$(<$($gen),*>)?;
            fn as_raw(&self) -> $crate::execution::RawKernelArguments {
                $crate::raw_kernel_arguments!($(&self.$arg_ident),*)
            }
        }
    };
}

#[macro_export]
macro_rules! cuda_kernel_signature_and_arguments {
    ($vis:vis $name:ident$(<$($gen:tt$(:$gen_tr:tt)?),+>)?, $($arg_ident:ident:$arg_ty:ty),*$(,)?) => {
        $crate::paste::paste! {
            $crate::cuda_kernel_signature!($vis [<$name Signature>]$(<$($gen),*>)?, $($arg_ident:$arg_ty),*);
            $crate::cuda_kernel_arguments!($vis [<$name Arguments>]$(<$($gen$(:$gen_tr)?),*>)?, [<$name Signature>], $($arg_ident:$arg_ty),*);
        }
    };
}

#[macro_export]
macro_rules! cuda_kernel_function {
    ($vis:vis $name:ident$(<$($gen:tt$(:$gen_tr:tt)?),+>)?, $signature_name:ident) => {
        $vis struct $name$(<$($gen$(:$gen_tr)?),*>)?($signature_name$(<$($gen),*>)?);
        impl$(<$($gen$(:$gen_tr)?),*>)? $crate::execution::KernelFunction for $name$(<$($gen),*>)? {
            type Signature = $signature_name$(<$($gen),*>)?;
            fn as_ptr(&self) -> *const std::os::raw::c_void {
                self.0 as *const std::os::raw::c_void
            }
        }
    };
}

#[macro_export]
macro_rules! cuda_kernel_signature_arguments_and_function {
    ($vis:vis $name:ident$(<$($gen:tt$(:$gen_tr:tt)?),+>)?, $($arg_ident:ident:$arg_ty:ty),*$(,)?) => {
        $crate::paste::paste! {
            $crate::cuda_kernel_signature_and_arguments!($vis $name$(<$($gen$(:$gen_tr)?),*>)?, $($arg_ident:$arg_ty),*);
            $crate::cuda_kernel_function!($vis [<$name Function>]$(<$($gen$(:$gen_tr)?),*>)?, [<$name Signature>]);
        }
    };
}

#[macro_export]
macro_rules! cuda_kernel_declaration {
    ($vis:vis $kernel_name:ident($($arg_ident:ident:$arg_ty:ty),*$(,)?)) => {
        ::era_cudart_sys::cuda_fn_and_stub! {$vis fn $kernel_name($($arg_ident:$arg_ty,)*); }
    };
}

#[macro_export]
macro_rules! cuda_kernel {
    ($vis:vis $name:ident, $kernel_name:ident($($arg_ident:ident:$arg_ty:ty),*$(,)?)) => {
        $crate::paste::paste! {
            $crate::cuda_kernel_signature_arguments_and_function!($vis $name, $($arg_ident:$arg_ty),*);
            $crate::cuda_kernel_declaration!($vis $kernel_name($($arg_ident:$arg_ty),*));
            impl core::default::Default for [<$name Function>] {
                fn default() -> Self { Self($kernel_name) }
            }
        }
    };
    ($vis:vis $name:ident, $macro_name:ident, $($arg_ident:ident:$arg_ty:ty),*$(,)?) => {
        $crate::cuda_kernel_signature_arguments_and_function!($name,$($arg_ident:$arg_ty),*);
        macro_rules! $macro_name {
            ($kernel_name:ident) => {
                ::era_cudart::cuda_kernel_declaration!($kernel_name($($arg_ident:$arg_ty),*));
            };
        }
    };
}

pub struct HostFn<'a> {
    arc: Arc<Box<dyn Fn() + Send + Sync + 'a>>,
}

impl<'a> HostFn<'a> {
    pub fn new(func: impl Fn() + Send + Sync + 'a) -> Self {
        Self {
            arc: Arc::new(Box::new(func) as Box<dyn Fn() + Send + Sync>),
        }
    }
}

unsafe extern "C" fn launch_host_fn_callback(data: *mut c_void) {
    let raw = data as *const Box<dyn Fn() + Send + Sync>;
    let weak = Weak::from_raw(raw);
    if let Some(func) = weak.upgrade() {
        func();
    }
}

fn get_raw_fn_and_data(host_fn: &HostFn) -> (cudaHostFn_t, *mut c_void) {
    let weak = Arc::downgrade(&host_fn.arc);
    let raw = weak.into_raw();
    let data = raw as *mut c_void;
    (Some(launch_host_fn_callback), data)
}

pub fn launch_host_fn(stream: &CudaStream, host_fn: &HostFn) -> CudaResult<()> {
    let (func, data) = get_raw_fn_and_data(host_fn);
    unsafe { cudaLaunchHostFunc(stream.into(), func, data).wrap() }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn host_fn_add_executes_one_time() {
        let stream = CudaStream::create().unwrap();
        let mut a = 0;
        let add = || {
            a += 1;
            thread::sleep(Duration::from_millis(10));
        };
        let add_mutex = Mutex::new(add);
        let add_fn = HostFn::new(move || add_mutex.lock().unwrap()());
        let sleep_fn = HostFn::new(|| thread::sleep(Duration::from_millis(10)));
        launch_host_fn(&stream, &add_fn).unwrap();
        stream.synchronize().unwrap();
        launch_host_fn(&stream, &sleep_fn).unwrap();
        launch_host_fn(&stream, &add_fn).unwrap();
        drop(add_fn);
        stream.synchronize().unwrap();
        assert_eq!(a, 1);
    }
}
