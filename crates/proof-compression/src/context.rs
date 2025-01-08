use crate::{AsyncHandler, ProofSystemDefinition};

pub trait ContextInitializator {
    fn init<P>(config: P::ContextConfig) -> AsyncHandler<P::Context>
    where
        P: ProofSystemDefinition;
}

pub struct SimpelContextInitializor;

impl SimpelContextInitializor {
    pub fn new() -> Self {
        todo!()
    }
}
impl ContextInitializator for SimpelContextInitializor {
    fn init<P>(config: P::ContextConfig) -> AsyncHandler<P::Context>
    where
        P: ProofSystemDefinition,
    {
        todo!()
    }
}
