use std::sync::atomic::AtomicBool;

use crate::{AsyncHandler, CompressionProofSystem, SnarkWrapperProofSystem};

pub trait ContextManagerInterface {
    fn init_compression_context<P>(&self, config: P::ContextConfig) -> AsyncHandler<P::Context>
    where
        P: CompressionProofSystem;

    fn initialize_snark_context_config<S>(&self) -> AsyncHandler<S::ContextConfig>
    where
        S: SnarkWrapperProofSystem;

    fn init_snark_context<S>(
        &self,
        config: AsyncHandler<S::ContextConfig>,
    ) -> AsyncHandler<S::Context>
    where
        S: SnarkWrapperProofSystem;
}

pub struct SimpleContextManager {
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
    fn init_compression_context<P>(&self, config: P::ContextConfig) -> AsyncHandler<P::Context>
    where
        P: CompressionProofSystem,
    {
        assert!(
            self.context_status
                .load(std::sync::atomic::Ordering::Relaxed)
                == false
        );
        // load next context
        let flag = self.context_status.clone();
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let context = P::init_context(config);
            sender.send(context).unwrap();
            flag.store(false, std::sync::atomic::Ordering::Relaxed);
            receiver
        };

        AsyncHandler::spawn(f)
    }

    fn initialize_snark_context_config<S>(&self) -> AsyncHandler<S::ContextConfig>
    where
        S: SnarkWrapperProofSystem,
    {
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let start = std::time::Instant::now();
            let context_config = S::get_context_config();
            println!("CRS loading takes {}s", start.elapsed().as_secs());
            sender.send(context_config).unwrap();

            receiver
        };
        AsyncHandler::spawn(f)
    }

    fn init_snark_context<S>(
        &self,
        config: AsyncHandler<S::ContextConfig>,
    ) -> AsyncHandler<S::Context>
    where
        S: SnarkWrapperProofSystem,
    {
        assert!(
            self.context_status
                .load(std::sync::atomic::Ordering::Relaxed)
                == false
        );
        // load next context
        let flag = self.context_status.clone();
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let context = S::init_context(config.wait());
            sender.send(context).unwrap();
            flag.store(false, std::sync::atomic::Ordering::Relaxed);
            receiver
        };

        AsyncHandler::spawn(f)
    }
}
