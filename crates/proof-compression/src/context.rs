use crate::{AsyncHandler, ProofSystemDefinition};

pub trait ContextInitializator {
    fn init<P>(config: P::ContextConfig) -> AsyncHandler<P::Context>
    where
        P: ProofSystemDefinition;
}
