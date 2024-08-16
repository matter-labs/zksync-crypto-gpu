use crate::GpuTreeHasher;
use boojum::config::{
    CSConfig, CSSetupConfig, CSWitnessEvaluationConfig, DevCSConfig, ProvingCSConfig, SetupCSConfig,
};
use boojum::cs::cs_builder::new_builder;
use boojum::cs::cs_builder_reference::CsReferenceImplementationBuilder;
use boojum::cs::implementations::pow::NoPow;
use boojum::cs::implementations::proof::Proof;
use boojum::cs::implementations::prover::ProofConfig;
use boojum::cs::implementations::reference_cs::{CSReferenceAssembly, CSReferenceImplementation};
use boojum::cs::implementations::setup::FinalizationHintsForProver;
use boojum::cs::implementations::transcript::Transcript;
use boojum::cs::implementations::verifier::{VerificationKey, Verifier};
use boojum::cs::traits::GoodAllocator;
use boojum::cs::{CSGeometry, GateConfigurationHolder, StaticToolboxHolder};
use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};
use circuit_definitions::circuit_definitions::aux_layer::compression::{
    CompressionLayerCircuit, ProofCompressionFunction,
};
use circuit_definitions::circuit_definitions::aux_layer::{
    ZkSyncCompressionForWrapperCircuit, ZkSyncCompressionLayerCircuit,
};
use circuit_definitions::circuit_definitions::base_layer::ZkSyncBaseLayerCircuit;
use circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursiveLayerCircuit;
#[allow(unused_imports)]
use circuit_definitions::circuit_definitions::{
    ZkSyncUniformCircuitInstance, ZkSyncUniformSynthesisFunction,
};
use circuit_definitions::{
    base_layer_proof_config, recursion_layer_proof_config, ZkSyncDefaultRoundFunction,
};

type F = GoldilocksField;
type P = F;
#[allow(dead_code)]
type ZksyncProof<H> = Proof<F, H, GoldilocksExt2>;
#[allow(clippy::upper_case_acronyms)]
type EXT = GoldilocksExt2;

#[allow(clippy::large_enum_variant)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum CircuitWrapper {
    Base(ZkSyncBaseLayerCircuit),
    Recursive(ZkSyncRecursiveLayerCircuit),
    CompressionLayer(ZkSyncCompressionLayerCircuit),
    CompressionWrapper(ZkSyncCompressionForWrapperCircuit),
}

#[allow(dead_code)]
impl CircuitWrapper {
    pub fn geometry(&self) -> CSGeometry {
        match self {
            CircuitWrapper::Base(inner) => inner.geometry(),
            CircuitWrapper::Recursive(inner) => inner.geometry(),
            CircuitWrapper::CompressionLayer(inner) => inner.geometry(),
            CircuitWrapper::CompressionWrapper(inner) => inner.geometry(),
        }
    }
    pub fn size_hint(&self) -> (Option<usize>, Option<usize>) {
        match self {
            CircuitWrapper::Base(inner) => inner.size_hint(),
            CircuitWrapper::Recursive(inner) => inner.size_hint(),
            CircuitWrapper::CompressionLayer(inner) => inner.size_hint(),
            CircuitWrapper::CompressionWrapper(inner) => inner.size_hint(),
        }
    }

    pub fn numeric_circuit_type(&self) -> u8 {
        match self {
            CircuitWrapper::Base(inner) => inner.numeric_circuit_type(),
            CircuitWrapper::Recursive(inner) => inner.numeric_circuit_type(),
            CircuitWrapper::CompressionLayer(inner) => inner.numeric_circuit_type(),
            CircuitWrapper::CompressionWrapper(inner) => inner.numeric_circuit_type(),
        }
    }

    pub fn short_description(&self) -> &str {
        match self {
            CircuitWrapper::Base(inner) => inner.short_description(),
            CircuitWrapper::Recursive(inner) => inner.short_description(),
            CircuitWrapper::CompressionLayer(inner) => inner.short_description(),
            CircuitWrapper::CompressionWrapper(inner) => inner.short_description(),
        }
    }

    pub fn into_base_layer(self) -> ZkSyncBaseLayerCircuit {
        match self {
            CircuitWrapper::Base(inner) => inner,
            _ => unimplemented!(),
        }
    }

    pub fn into_recursive_layer(self) -> ZkSyncRecursiveLayerCircuit {
        match self {
            CircuitWrapper::Recursive(inner) => inner,
            _ => unimplemented!(),
        }
    }

    pub fn into_compression_layer(self) -> ZkSyncCompressionLayerCircuit {
        match self {
            CircuitWrapper::CompressionLayer(inner) => inner,
            _ => unimplemented!(),
        }
    }

    pub fn into_compression_wrapper(self) -> ZkSyncCompressionForWrapperCircuit {
        match self {
            CircuitWrapper::CompressionWrapper(inner) => inner,
            _ => unimplemented!(),
        }
    }

    pub fn as_base_layer(&self) -> &ZkSyncBaseLayerCircuit {
        match self {
            CircuitWrapper::Base(inner) => inner,
            _ => unimplemented!(),
        }
    }

    pub fn as_recursive_layer(&self) -> &ZkSyncRecursiveLayerCircuit {
        match self {
            CircuitWrapper::Recursive(inner) => inner,
            _ => unimplemented!(),
        }
    }

    pub fn is_base_layer(&self) -> bool {
        matches!(self, CircuitWrapper::Base(_))
    }

    pub fn proof_config(&self) -> ProofConfig {
        match self {
            CircuitWrapper::Base(_) => base_layer_proof_config(),
            CircuitWrapper::Recursive(_) => recursion_layer_proof_config(),
            CircuitWrapper::CompressionLayer(compression_circuit) => {
                compression_circuit.proof_config_for_compression_step()
            }
            CircuitWrapper::CompressionWrapper(compression_wrapper_circuit) => {
                compression_wrapper_circuit.proof_config_for_compression_step()
            }
        }
    }

    pub fn verify_proof<T: Transcript<F>, H: GpuTreeHasher<Output = T::CompatibleCap>>(
        &self,
        transcript_params: T::TransciptParameters,
        vk: &VerificationKey<F, H>,
        proof: &ZksyncProof<H>,
    ) -> bool {
        let verifier = self.get_verifier();
        verifier.verify::<H, T, NoPow>(transcript_params, vk, proof)
    }

    pub(crate) fn get_verifier(&self) -> Verifier<F, EXT> {
        match self {
            CircuitWrapper::Base(inner) => get_verifier_for_base_layer_circuit(inner),
            CircuitWrapper::Recursive(inner) => get_verifier_for_recursive_layer_circuit(inner),
            CircuitWrapper::CompressionLayer(inner) => {
                get_verifier_for_compression_layer_circuit(inner)
            }
            CircuitWrapper::CompressionWrapper(inner) => {
                get_verifier_for_compression_wrapper_circuit(inner)
            }
        }
    }
}

pub(crate) fn get_verifier_for_base_layer_circuit(
    circuit: &ZkSyncBaseLayerCircuit,
) -> Verifier<F, EXT> {
    use circuit_definitions::circuit_definitions::verifier_builder::dyn_verifier_builder_for_circuit_type;
    let verifier_builder = dyn_verifier_builder_for_circuit_type(circuit.numeric_circuit_type());
    verifier_builder.create_verifier()
}

pub(crate) fn get_verifier_for_recursive_layer_circuit(
    circuit: &ZkSyncRecursiveLayerCircuit,
) -> Verifier<F, EXT> {
    let verifier_builder = circuit.into_dyn_verifier_builder();
    verifier_builder.create_verifier()
}

pub(crate) fn get_verifier_for_compression_layer_circuit(
    circuit: &ZkSyncCompressionLayerCircuit,
) -> Verifier<F, EXT> {
    let verifier_builder = circuit.into_dyn_verifier_builder();
    verifier_builder.create_verifier()
}

pub(crate) fn get_verifier_for_compression_wrapper_circuit(
    circuit: &ZkSyncCompressionForWrapperCircuit,
) -> Verifier<F, EXT> {
    let verifier_builder = circuit.into_dyn_verifier_builder();
    verifier_builder.create_verifier()
}

#[allow(dead_code)]
pub(crate) fn synth_circuit_for_setup(
    circuit: CircuitWrapper,
) -> (
    CSReferenceAssembly<F, P, SetupCSConfig>,
    FinalizationHintsForProver,
) {
    let (cs, some_finalization_hint) = init_or_synthesize_assembly::<_, true>(circuit, None);
    assert!(cs.next_available_place_idx() > 0);
    (cs, some_finalization_hint.expect("finalization hint"))
}

#[allow(dead_code)]
pub(crate) fn synth_circuit_for_proving(
    circuit: CircuitWrapper,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    let (cs, some_finalization_hint) = init_or_synthesize_assembly::<_, true>(circuit, Some(hint));
    assert!(some_finalization_hint.is_none());
    assert!(cs.next_available_place_idx() > 0);
    cs
}

// called by zksync-era
pub fn init_base_layer_cs_for_repeated_proving(
    circuit: ZkSyncBaseLayerCircuit,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    init_cs_for_external_proving(CircuitWrapper::Base(circuit), hint)
}

// called by zksync-era
pub fn init_recursive_layer_cs_for_repeated_proving(
    circuit: ZkSyncRecursiveLayerCircuit,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    init_cs_for_external_proving(CircuitWrapper::Recursive(circuit), hint)
}

pub(crate) fn init_cs_for_external_proving(
    circuit: CircuitWrapper,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    let (cs, some_finalization_hint) = init_or_synthesize_assembly::<_, false>(circuit, Some(hint));
    assert!(some_finalization_hint.is_none());
    assert_eq!(cs.next_available_place_idx(), 0);
    cs
}

// in init_or_synthesize_assembly, we expect CFG to be either
// ProvingCSConfig or SetupCSConfig
pub trait AllowInitOrSynthesize: CSConfig {}

impl AllowInitOrSynthesize for ProvingCSConfig {}

impl AllowInitOrSynthesize for SetupCSConfig {}

impl AllowInitOrSynthesize for DevCSConfig {}

pub(crate) fn init_or_synthesize_assembly<CFG: AllowInitOrSynthesize, const DO_SYNTH: bool>(
    circuit: CircuitWrapper,
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG>,
    Option<FinalizationHintsForProver>,
) {
    let geometry = circuit.geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl =
        CsReferenceImplementationBuilder::<F, P, CFG>::new(geometry, max_trace_len.unwrap());
    let builder = new_builder::<_, F>(builder_impl);
    let round_function = ZkSyncDefaultRoundFunction::default();

    // if we are proving then we need finalization hint
    assert_eq!(
        finalization_hint.is_some(),
        <CFG::WitnessConfig as CSWitnessEvaluationConfig>::EVALUATE_WITNESS
    );
    assert_eq!(
        finalization_hint.is_none(),
        <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP
    );
    // if we are just creating reusable assembly then cs shouldn't be configured for setup
    if !DO_SYNTH {
        assert!(!<CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP);
        assert!(<CFG::WitnessConfig as CSWitnessEvaluationConfig>::EVALUATE_WITNESS);
    }

    let builder_arg = num_vars.unwrap();

    match circuit {
        CircuitWrapper::Base(base_circuit) => match base_circuit {
            ZkSyncBaseLayerCircuit::MainVM(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::CodeDecommittmentsSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::CodeDecommitter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::LogDemuxer(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::KeccakRoundFunction(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::Sha256RoundFunction(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::ECRecover(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::RAMPermutation(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::StorageSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::StorageApplication(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::EventsSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::L1MessagesSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::L1MessagesHasher(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::TransientStorageSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::Secp256r1Verify(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::EIP4844Repack(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
        },
        CircuitWrapper::Recursive(recursive_circuit) => match recursive_circuit {
            ZkSyncRecursiveLayerCircuit::SchedulerCircuit(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncRecursiveLayerCircuit::NodeLayerCircuit(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForMainVM(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForCodeDecommittmentsSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForCodeDecommitter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForLogDemuxer(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForKeccakRoundFunction(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForSha256RoundFunction(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForECRecover(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForRAMPermutation(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForStorageSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForStorageApplication(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForEventsSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForL1MessagesSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForL1MessagesHasher(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForTransientStorageSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForSecp256r1Verify(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForEIP4844Repack(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncRecursiveLayerCircuit::RecursionTipCircuit(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
        },
        CircuitWrapper::CompressionLayer(compression_circuit) => match compression_circuit {
            ZkSyncCompressionLayerCircuit::CompressionMode1Circuit(inner) => {
                synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
            }
            ZkSyncCompressionLayerCircuit::CompressionMode2Circuit(inner) => {
                synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
            }
            ZkSyncCompressionLayerCircuit::CompressionMode3Circuit(inner) => {
                synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
            }
            ZkSyncCompressionLayerCircuit::CompressionMode4Circuit(inner) => {
                synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
            }
            ZkSyncCompressionLayerCircuit::CompressionMode5Circuit(inner) => {
                synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
            }
        },
        CircuitWrapper::CompressionWrapper(compression_wrapper_circuit) => {
            match compression_wrapper_circuit {
                ZkSyncCompressionForWrapperCircuit::CompressionMode1Circuit(inner) => {
                    synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
                }
                ZkSyncCompressionForWrapperCircuit::CompressionMode2Circuit(inner) => {
                    synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
                }
                ZkSyncCompressionForWrapperCircuit::CompressionMode3Circuit(inner) => {
                    synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
                }
                ZkSyncCompressionForWrapperCircuit::CompressionMode4Circuit(inner) => {
                    synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
                }
                ZkSyncCompressionForWrapperCircuit::CompressionMode5Circuit(inner) => {
                    synthesize_compression_circuit(inner, DO_SYNTH, finalization_hint)
                }
            }
        }
    }
}

fn into_assembly<
    CFG: CSConfig,
    GC: GateConfigurationHolder<F>,
    T: StaticToolboxHolder,
    A: GoodAllocator,
>(
    mut cs: CSReferenceImplementation<F, P, CFG, GC, T>,
    do_synth: bool,
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG, A>,
    Option<FinalizationHintsForProver>,
) {
    if <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP {
        let (_, finalization_hint) = cs.pad_and_shrink();
        (cs.into_assembly(), Some(finalization_hint))
    } else {
        let hint = finalization_hint.unwrap();
        if do_synth {
            cs.pad_and_shrink_using_hint(hint);
            (cs.into_assembly(), None)
        } else {
            (cs.into_assembly_for_repeated_proving(hint), None)
        }
    }
}

pub fn synthesize_compression_circuit<
    CF: ProofCompressionFunction,
    CFG: CSConfig,
    A: GoodAllocator,
>(
    circuit: CompressionLayerCircuit<CF>,
    do_synth: bool,
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG, A>,
    Option<FinalizationHintsForProver>,
) {
    let geometry = circuit.geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GoldilocksField, F, CFG>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GoldilocksField>(builder_impl);

    let builder = circuit.configure_builder_proxy(builder);
    let mut cs = builder.build(num_vars.unwrap());
    circuit.add_tables(&mut cs);
    if do_synth {
        circuit.synthesize_into_cs(&mut cs);
    }

    into_assembly(cs, do_synth, finalization_hint)
}
