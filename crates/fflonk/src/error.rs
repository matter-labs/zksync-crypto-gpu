#[derive(Clone, Debug)]
pub enum CudaError {
    Error(String),
    VariableAssignmentError(String),
    MaterializePermutationsError(String),
    SetupError(String),
    NttError(String),
    MsmError(String),
    TransferError(String),
    AllocationError(String),
    SyncError(String),
}
pub type CudaResult<T> = Result<T, CudaError>;
