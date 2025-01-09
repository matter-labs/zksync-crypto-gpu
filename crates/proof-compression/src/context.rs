use std::sync::atomic::AtomicBool;

use crate::{AsyncHandler, ProofSystemDefinition};

pub trait ContextManagerInterface {
    fn init_context<P>(&self) -> AsyncHandler<P::Context>
    where
        P: ProofSystemDefinition;
}

pub struct SimpleContextManager(std::sync::Arc<AtomicBool>);

impl SimpleContextManager {
    pub fn new() -> Self {
        SimpleContextManager(std::sync::Arc::new(AtomicBool::new(false)))
    }
}
impl ContextManagerInterface for SimpleContextManager {
    fn init_context<P>(&self) -> AsyncHandler<P::Context>
    where
        P: ProofSystemDefinition,
    {
        assert!(self.0.load(std::sync::atomic::Ordering::Relaxed) == false);
        // load next context
        let flag = self.0.clone();
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let config = P::get_context_config();
            let context = P::init_context(config);
            // mark status
            flag.store(false, std::sync::atomic::Ordering::Relaxed);
            sender.send(context).unwrap();

            receiver
        };

        AsyncHandler::spawn(f)
    }
}
