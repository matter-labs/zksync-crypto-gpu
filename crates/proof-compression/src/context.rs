use std::sync::atomic::AtomicBool;

use crate::{CompressionProofSystem, SnarkWrapperProofSystem, SnarkWrapperStep};

pub(crate) trait ContextManagerInterface {
    fn init_compression_context<P>(&self, config: P::ContextConfig) -> P::Context
    where
        P: CompressionProofSystem;

    fn init_snark_context<S>(&self, crs: S::CRS) -> S::Context
    where
        S: SnarkWrapperStep;
}

pub(crate) struct SimpleContextManager {
    context_status: std::sync::Arc<AtomicBool>,
}

impl SimpleContextManager {
    pub fn new() -> Self {
        Self {
            context_status: std::sync::Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ContextManagerInterface for SimpleContextManager {
    fn init_compression_context<P>(&self, config: P::ContextConfig) -> P::Context
    where
        P: CompressionProofSystem,
    {
        assert!(
            self.context_status
                .load(std::sync::atomic::Ordering::Relaxed)
                == false
        );
        // let flag = self.context_status.clone();
        // let f = move || {
        //     let (sender, receiver) = std::sync::mpsc::channel();
        //     let context = P::init_context(config);
        //     sender.send(context).unwrap();
        //     flag.store(false, std::sync::atomic::Ordering::Relaxed);
        //     receiver
        // };

        // AsyncHandler::spawn(f)

        P::init_context(config)
    }
    fn init_snark_context<S>(&self, compact_raw_crs: S::CRS) -> S::Context
    where
        S: SnarkWrapperProofSystem,
    {
        assert!(
            self.context_status
                .load(std::sync::atomic::Ordering::Relaxed)
                == false
        );
        // load next context
        // let flag = self.context_status.clone();
        // let f = move || {
        //     let (sender, receiver) = std::sync::mpsc::channel();
        //     let context = S::init_context(compact_raw_crs);
        //     sender.send(context).unwrap();
        //     flag.store(false, std::sync::atomic::Ordering::Relaxed);
        //     receiver
        // };

        // AsyncHandler::spawn(f)

        S::init_context(compact_raw_crs)
    }
}
