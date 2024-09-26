use bellman::bn256::Bn256;
use bellman::kate_commitment::CrsForMonomialForm;
use bellman::CurveAffine;
use boojum::algebraic_props::round_function::AbsorptionModeOverwrite;
use boojum::algebraic_props::sponge::GoldilocksPoseidon2Sponge;
use boojum::config::{ProvingCSConfig, SetupCSConfig};
use boojum::cs::implementations::proof::Proof;
use boojum::cs::implementations::prover::ProofConfig;
use boojum::cs::implementations::{transcript::GoldilocksPoisedon2Transcript, verifier::Verifier};
use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};
use boojum::worker::Worker;
use shivini::circuit_definitions::boojum;
use shivini::circuit_definitions::circuit_definitions::{
    aux_layer::{
        compression::{CompressionLayerCircuit, ProofCompressionFunction},
        compression_modes::{CompressionTranscriptForWrapper, CompressionTreeHasherForWrapper},
        CompressionProofsTreeHasher, CompressionProofsTreeHasherForWrapper,
        ZkSyncCompressionForWrapperCircuit, ZkSyncCompressionLayerCircuit, ZkSyncCompressionProof,
        ZkSyncCompressionProofForWrapper, ZkSyncCompressionVerificationKey,
        ZkSyncCompressionVerificationKeyForWrapper,
    },
    recursion_layer::{
        ZkSyncRecursionLayerProof, ZkSyncRecursionLayerVerificationKey, ZkSyncRecursionProof,
        ZkSyncRecursionVerificationKey,
    },
};
use shivini::cs::{GpuProverSetupData, GpuSetup};
use shivini::gpu_proof_config::GpuProofConfig;
use shivini::synthesis_utils::synthesize_compression_circuit;
use shivini::{
    gpu_prove_from_external_witness_data_with_cache_strategy, CacheStrategy,
    CommitmentCacheStrategy, PolynomialsCacheStrategy, ProverContext, ProverContextConfig,
};
use std::alloc::Global;
use std::io::Read;

use fflonk::*;

type F = GoldilocksField;
type EXT = GoldilocksExt2;
type DefaultTreeHasher = GoldilocksPoseidon2Sponge<AbsorptionModeOverwrite>;
type DefaultTranscript = GoldilocksPoisedon2Transcript;

pub type ZksyncProof = Proof<F, DefaultTreeHasher, EXT>;
type CompressionProofsTranscript = GoldilocksPoisedon2Transcript;

fn load_scheduler_proof_and_vk(
    path: &str,
) -> (ZkSyncRecursionProof, ZkSyncRecursionVerificationKey) {
    let scheduler_vk_file =
        std::fs::File::open(format!("{}/scheduler_recursive_vk.json", path)).unwrap();
    let scheduler_vk: ZkSyncRecursionLayerVerificationKey =
        serde_json::from_reader(&scheduler_vk_file).unwrap();
    let scheduler_proof_file =
        std::fs::File::open(format!("{}/scheduler_recursive_proof.json", path)).unwrap();
    let scheduler_proof: ZkSyncRecursionLayerProof =
        serde_json::from_reader(&scheduler_proof_file).unwrap();

    (scheduler_proof.into_inner(), scheduler_vk.into_inner())
}

#[test]
fn download_crs() {
    let degree = MAX_COMBINED_DEGREE_FACTOR << L1_VERIFIER_DOMAIN_SIZE_LOG;
    download_and_transform_ignition_transcripts(degree);
}

#[test]
fn transform_crs() {
    use bellman::compact_bn256::Bn256 as CompactBn256;
    use bellman::compact_bn256::G1Affine as CompactG1Affine;
    use bellman::compact_bn256::G2Affine as CompactG2Affine;
    use bellman::kate_commitment::Crs;
    let transcripts_dir = std::env::var("IGNITION_TRANSCRIPT_PATH").unwrap_or("./".to_string());
    let crs_path = format!("{}/full_ignition.key", &transcripts_dir);
    println!("Loading original CRS file");
    let crs_file = std::fs::File::open(&crs_path).unwrap();
    let original_crs = Crs::<Bn256, CrsForMonomialForm>::read(&crs_file).unwrap();

    let Crs {
        g1_bases,
        g2_monomial_bases,
        ..
    } = original_crs;
    let g1_bases = std::sync::Arc::try_unwrap(g1_bases).unwrap();
    let g2_bases = std::sync::Arc::try_unwrap(g2_monomial_bases).unwrap();
    println!("Transforming G1 bases");
    let mut transformed_g1 = vec![];
    for p in g1_bases {
        let (x, y) = p.as_xy();
        let compact_point = CompactG1Affine::from_xy_unchecked(x.clone(), y.clone());
        transformed_g1.push(compact_point);
    }

    println!("Transforming G2 bases");
    let mut transformed_g2 = vec![];
    for p in g2_bases {
        let (x, y) = p.as_xy();
        let compact_point = CompactG2Affine::from_xy_unchecked(x.clone(), y.clone());
        transformed_g2.push(compact_point);
    }

    let transformed_crs =
        Crs::<CompactBn256, CrsForMonomialForm>::new(transformed_g1, transformed_g2);
    println!("Saving transformed CRS");
    let new_crs_file_path = format!("{}/full_ignition_compact.key", transcripts_dir);
    let new_crs_file = std::fs::File::create(&new_crs_file_path).unwrap();
    transformed_crs.write(&new_crs_file).unwrap();
    println!("Transformed CRS saved into {}", new_crs_file_path);
}

#[derive(Copy, Clone, Debug)]
pub enum CompressionMode {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
}

impl CompressionMode {
    pub fn from_compression_mode(compression_mode: u8) -> Self {
        match compression_mode {
            1 => CompressionMode::One,
            2 => CompressionMode::Two,
            3 => CompressionMode::Three,
            4 => CompressionMode::Four,
            5 => CompressionMode::Five,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct CompressionSchedule {
    name: &'static str,
    pub compression_steps: Vec<CompressionMode>,
}

impl CompressionSchedule {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn hard() -> Self {
        CompressionSchedule {
            name: "hard",
            compression_steps: vec![
                CompressionMode::One,
                CompressionMode::Two,
                CompressionMode::Three,
                CompressionMode::Four,
            ],
        }
    }
}

pub enum CompressionInput {
    Recursion(
        Option<ZkSyncRecursionProof>,
        ZkSyncRecursionVerificationKey,
        CompressionMode,
    ),
    Compression(
        Option<ZkSyncCompressionProof>,
        ZkSyncCompressionVerificationKey,
        CompressionMode,
    ),
    CompressionWrapper(
        Option<ZkSyncCompressionProof>,
        ZkSyncCompressionVerificationKey,
        CompressionMode,
    ),
}

impl CompressionInput {
    pub fn into_compression_circuit(self) -> ZkSyncCompressionLayerCircuit {
        match self {
            CompressionInput::Recursion(proof, vk, compression_mode) => {
                assert_eq!(compression_mode as u8, 1);
                ZkSyncCompressionLayerCircuit::from_witness_and_vk(proof, vk, 1)
            }
            CompressionInput::Compression(proof, vk, compression_mode) => {
                ZkSyncCompressionLayerCircuit::from_witness_and_vk(
                    proof,
                    vk,
                    compression_mode as u8,
                )
            }
            CompressionInput::CompressionWrapper(_, _, _) => {
                unreachable!()
            }
        }
    }

    pub fn into_compression_wrapper_circuit(self) -> ZkSyncCompressionForWrapperCircuit {
        match self {
            CompressionInput::Recursion(_, _, _) => {
                unreachable!()
            }
            CompressionInput::Compression(_, _, _) => {
                unreachable!()
            }
            CompressionInput::CompressionWrapper(proof, vk, compression_mode) => {
                ZkSyncCompressionForWrapperCircuit::from_witness_and_vk(
                    proof,
                    vk,
                    compression_mode as u8,
                )
            }
        }
    }
}

#[test]
fn run_proof_compression_by_schedule() {
    let path = if let Ok(path) = std::env::var("BLOB_PATH") {
        path.to_string()
    } else {
        "./data".to_string()
    };
    let (scheduler_proof, scheduler_vk) = load_scheduler_proof_and_vk(&path);
    process_steps(
        scheduler_proof,
        scheduler_vk,
        CompressionSchedule::hard(),
        &path,
    );
}

pub fn process_steps(
    proof: ZkSyncRecursionProof,
    vk: ZkSyncRecursionVerificationKey,
    schedule: CompressionSchedule,
    path: &str,
) {
    let worker = Worker::new();
    let mut input = CompressionInput::Recursion(Some(proof), vk, CompressionMode::One);

    dbg!(&schedule);
    let CompressionSchedule {
        name: compression_schedule_name,
        compression_steps,
    } = schedule;

    let last_compression_wrapping_mode =
        CompressionMode::from_compression_mode(*compression_steps.last().unwrap() as u8 + 1);
    dbg!(&last_compression_wrapping_mode);

    /*
        This illustrates how compression enforced for the "hardest" strategy

           input                       compression     verifier          output        compression wrapper
       _____________________________   ____________    ___________     __________      ___________________
       scheduler       proof   vk          1           scheduler   ->  compressed1         compressed2
       compressed1     proof   vk          2           compressed1 ->  compressed2         compressed3
       compressed2     proof   vk          3           compressed2 ->  compressed3         compressed4
       compressed3     proof   vk          4           compressed3 ->  compressed4         compressed5


       compressed5     proof   vk          -       compression wrapper5       ->  fflonk proof
    */

    let num_compression_steps = compression_steps.len();
    let mut compression_modes_iter = compression_steps.into_iter();
    for step_idx in 0..num_compression_steps {
        let compression_mode = compression_modes_iter.next().unwrap();
        let proof_file_path = format!("{}/compression_{}_proof.json", path, compression_mode as u8);
        let proof_file_path = std::path::Path::new(&proof_file_path);
        let vk_file_path = format!("{}/compression_{}_vk.json", path, compression_mode as u8);
        let vk_file_path = std::path::Path::new(&vk_file_path);
        let setup_data_file_path = format!(
            "{}/compression_{}_setup_data.bin",
            path, compression_mode as u8
        );
        let setup_data_file_path = std::path::Path::new(&setup_data_file_path);
        if proof_file_path.exists() && vk_file_path.exists() {
            println!(
                "Compression {compression_schedule_name}/{} proof and vk already exist ignoring",
                compression_mode as u8
            );
            let proof_file = std::fs::File::open(proof_file_path).unwrap();
            let input_proof = serde_json::from_reader(&proof_file).unwrap();
            let vk_file = std::fs::File::open(vk_file_path).unwrap();
            let input_vk = serde_json::from_reader(&vk_file).unwrap();
            if step_idx + 1 == num_compression_steps {
                input = CompressionInput::CompressionWrapper(
                    input_proof,
                    input_vk,
                    last_compression_wrapping_mode,
                )
            } else {
                input = CompressionInput::Compression(
                    input_proof,
                    input_vk,
                    CompressionMode::from_compression_mode(compression_mode as u8 + 1),
                )
            }

            continue;
        }
        let mut setup_data = if setup_data_file_path.exists() {
            let bytes = std::fs::read(setup_data_file_path).unwrap();
            println!(
                "Compression wrapper setup data for {compression_schedule_name}/{} loaded",
                compression_mode as u8
            );
            Some(bincode::deserialize(&bytes).unwrap())
        } else {
            None
        };

        let compression_circuit = input.into_compression_circuit();
        let circuit_type = compression_circuit.numeric_circuit_type();
        println!(
            "Proving compression {compression_schedule_name}/{}",
            compression_mode as u8
        );
        let (output_proof, output_vk) =
            prove_compression_layer_circuit(compression_circuit.clone(), &mut setup_data, &worker);
        println!(
            "Proof for compression {compression_schedule_name}/{} is generated!",
            compression_mode as u8
        );

        save_compression_proof_and_vk_into_file(&output_proof, &output_vk, circuit_type, path);

        if setup_data.is_some() {
            let bytes = bincode::serialize(&setup_data.unwrap()).unwrap();
            std::fs::write(setup_data_file_path, bytes).unwrap();
            println!(
                "Compression wrapper setup data for {compression_schedule_name}/{} saved",
                compression_mode as u8
            );
        }

        if step_idx + 1 == num_compression_steps {
            input = CompressionInput::CompressionWrapper(
                Some(output_proof),
                output_vk,
                last_compression_wrapping_mode,
            );
        } else {
            input = CompressionInput::Compression(
                Some(output_proof),
                output_vk,
                CompressionMode::from_compression_mode(compression_mode as u8 + 1),
            );
        }
    }

    // last wrapping step
    let proof_file_path = format!(
        "{}/compression_wrapper_{}_proof.json",
        path, last_compression_wrapping_mode as u8
    );
    let compression_wrapper_proof_file_path = std::path::Path::new(&proof_file_path);
    let vk_file_path = format!(
        "{}/compression_wrapper_{}_vk.json",
        path, last_compression_wrapping_mode as u8
    );
    let compression_wrapper_vk_file_path = std::path::Path::new(&vk_file_path);
    let setup_data_file_path = format!(
        "{}/compression_wrapper_{}_setup_data.bin",
        path, last_compression_wrapping_mode as u8
    );
    let setup_data_file_path = std::path::Path::new(&setup_data_file_path);
    println!(
        "Compression for wrapper level {}",
        last_compression_wrapping_mode as u8
    );
    if compression_wrapper_proof_file_path.exists() && compression_wrapper_vk_file_path.exists() {
        println!(
            "Compression {compression_schedule_name}/{} for wrapper proof and vk already exist ignoring",
            last_compression_wrapping_mode as u8
        );
    } else {
        println!(
            "Proving compression {compression_schedule_name}/{} for wrapper",
            last_compression_wrapping_mode as u8
        );
        let mut setup_data = if setup_data_file_path.exists() {
            let bytes = std::fs::read(setup_data_file_path).unwrap();
            println!(
                "Compression wrapper setup data for {compression_schedule_name}/{} loaded",
                last_compression_wrapping_mode as u8
            );
            Some(bincode::deserialize(&bytes).unwrap())
        } else {
            None
        };
        let compression_circuit = input.into_compression_wrapper_circuit();
        let (compression_wrapper_output_proof, compression_wrapper_output_vk) =
            prove_compression_wrapper_circuit(compression_circuit, &mut setup_data, &worker);
        println!(
            "Proof for compression wrapper {compression_schedule_name}/{} is generated!",
            last_compression_wrapping_mode as u8
        );
        save_compression_wrapper_proof_and_vk_into_file(
            &compression_wrapper_output_proof,
            &compression_wrapper_output_vk,
            last_compression_wrapping_mode as u8,
            path,
        );
        println!(
            "Compression wrapper proof and vk for {compression_schedule_name}/{} saved",
            last_compression_wrapping_mode as u8
        );
        if setup_data.is_some() {
            let bytes = bincode::serialize(&setup_data.unwrap()).unwrap();
            std::fs::write(setup_data_file_path, bytes).unwrap();
            println!(
                "Compression wrapper setup data for {compression_schedule_name}/{} saved",
                last_compression_wrapping_mode as u8
            );
        }
    }
    // final wrapping step
    let final_proof_file_path = format!("{}/final_proof.json", path);
    let final_proof_file_path = std::path::Path::new(&final_proof_file_path);
    let final_vk_file_path = format!("{}/final_vk.json", path,);
    let final_vk_file_path = std::path::Path::new(&final_vk_file_path);

    if final_proof_file_path.exists() == false || final_vk_file_path.exists() == false {
        let (compression_wrapper_proof, compression_wrapper_vk) =
            load_compression_wrapper_proof_and_vk_from_file(
                path,
                last_compression_wrapping_mode as u8,
            );
        let wrapper_circuit = init_snark_wrapper_circuit_from_inputs(
            last_compression_wrapping_mode as u8,
            compression_wrapper_proof,
            compression_wrapper_vk,
        );

        let worker = bellman::worker::Worker::new();
        let setup_file_path = format!("{}/final_snark_device_setup.bin", path);
        let setup_file_path = std::path::Path::new(&setup_file_path);
        if setup_file_path.exists() == false {
            precompute_and_save_setup_and_vk_for_fflonk_snark_circuit(
                &wrapper_circuit,
                &worker,
                path,
            );
        } else {
            println!("device setup of fflonk already exist, loading");
        }

        // let (device_setup, final_vk) = load_device_setup_and_vk_of_fflonk_snark_circuit(path);
        let (device_setup, final_vk) =
            load_device_setup_in_memory_and_vk_of_fflonk_snark_circuit(path);
        let final_proof = fflonk::gpu_prove_fflonk_snark_verifier_circuit_with_precomputation(
            &wrapper_circuit,
            &device_setup,
            &final_vk,
            &worker,
        );
        let final_proof_file = std::fs::File::create(final_proof_file_path).unwrap();
        serde_json::to_writer(&final_proof_file, &final_proof).unwrap();
        println!(
            "final snark proof saved into {}",
            final_proof_file_path.to_string_lossy()
        );
        // save_fflonk_proof_and_vk_into_file(&final_proof, &final_vk, &path);
    } else {
        println!(
            "final proof already exists {}",
            final_proof_file_path.to_string_lossy()
        );
    }
}

#[test]
fn load_setup_into_memory() -> std::io::Result<()> {
    // This is convenient for faster testing
    let socket_path = std::env::var("TEST_SOCK_FILE").unwrap();
    let _ = std::fs::remove_file(&socket_path);

    let listener = std::os::unix::net::UnixListener::bind(&socket_path)?;

    println!("Loading setup data into memory");
    let setup_file_path = std::env::var("SETUP_FILE").unwrap();
    let mut setup_file = std::fs::File::open(&setup_file_path).unwrap();
    let mut setup_encoding = Vec::new();
    setup_file.read_to_end(&mut setup_encoding).unwrap();

    println!("Loading CRS data into memory");
    let crs_file_path = std::env::var("CRS_FILE").unwrap();
    let mut crs_file = std::fs::File::open(&crs_file_path).unwrap();
    let mut crs_encoding = Vec::new();
    crs_file.read_to_end(&mut crs_encoding).unwrap();

    println!("Server is listening on the socket: {}", socket_path);
    for stream in listener.incoming() {
        match stream {
            Ok(mut socket) => {
                println!("Client connected!");

                // Read data sent by the client
                let mut buffer = [0; 1024];
                let bytes_read = std::io::Read::read(&mut socket, &mut buffer)?;

                if buffer[0] == 0 {
                    println!("Sending setup encoding");
                    std::io::Write::write_all(&mut socket, &setup_encoding)?;
                } else if buffer[0] == 1 {
                    println!("Sending CRS encoding");
                    std::io::Write::write_all(&mut socket, &crs_encoding)?;
                } else {
                    println!("Unknowon request");
                }
            }
            Err(err) => {
                eprintln!("Error accepting connection: {}", err);
            }
        }
    }

    Ok(())
}

pub fn load_device_setup_in_memory_and_vk_of_fflonk_snark_circuit(
    path: &str,
) -> (
    FflonkSnarkVerifierCircuitDeviceSetup,
    FflonkSnarkVerifierCircuitVK,
) {
    println!("Loading fflonk setup for snark circuit");
    let device_setup = read_setup_over_socket().unwrap();
    let vk_file_path = format!("{}/final_vk.json", path);
    let vk_file_path = std::path::Path::new(&vk_file_path);
    let vk_file = std::fs::File::open(&vk_file_path).unwrap();
    let vk = serde_json::from_reader(&vk_file).unwrap();

    (device_setup, vk)
}

fn read_setup_over_socket() -> std::io::Result<FflonkSnarkVerifierCircuitDeviceSetup> {
    let socket_path = std::env::var("TEST_SOCK_FILE").unwrap();
    let mut socket = std::os::unix::net::UnixStream::connect(socket_path)?;
    let start = std::time::Instant::now();
    std::io::Write::write_all(&mut socket, &[0])?;
    let device_setup = FflonkSnarkVerifierCircuitDeviceSetup::read(&socket).unwrap();
    println!("Decoding setup takes {} s", start.elapsed().as_secs());

    Ok(device_setup)
}

pub fn save_compression_proof_and_vk_into_file(
    proof: &ZkSyncCompressionProof,
    vk: &ZkSyncCompressionVerificationKey,
    compression_mode: u8,
    path: &str,
) {
    let proof_file = std::fs::File::create(format!(
        "{}/compression_{}_proof.json",
        path, compression_mode
    ))
    .unwrap();
    serde_json::to_writer(proof_file, &proof).unwrap();
    let vk_file =
        std::fs::File::create(format!("{}/compression_{}_vk.json", path, compression_mode))
            .unwrap();
    serde_json::to_writer(vk_file, &vk).unwrap();
}

pub fn save_compression_wrapper_proof_and_vk_into_file(
    proof: &ZkSyncCompressionProofForWrapper,
    vk: &ZkSyncCompressionVerificationKeyForWrapper,
    compression_mode: u8,
    path: &str,
) {
    let proof_file = std::fs::File::create(format!(
        "{}/compression_wrapper_{}_proof.json",
        path, compression_mode
    ))
    .unwrap();
    serde_json::to_writer(proof_file, &proof).unwrap();
    let vk_file = std::fs::File::create(format!(
        "{}/compression_wrapper_{}_vk.json",
        path, compression_mode
    ))
    .unwrap();
    serde_json::to_writer(vk_file, &vk).unwrap();
}

pub fn load_compression_wrapper_proof_and_vk_from_file(
    blob_path: &str,
    compression_mode: u8,
) -> (
    ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
) {
    let proof_file = std::fs::File::open(format!(
        "{}/compression_wrapper_{}_proof.json",
        blob_path, compression_mode
    ))
    .unwrap();
    let proof = serde_json::from_reader(proof_file).unwrap();
    let vk_file = std::fs::File::open(format!(
        "{}/compression_wrapper_{}_vk.json",
        blob_path, compression_mode
    ))
    .unwrap();
    let vk = serde_json::from_reader(vk_file).unwrap();

    (proof, vk)
}

pub fn prove_compression_layer_circuit(
    circuit: ZkSyncCompressionLayerCircuit,
    setup_data: &mut Option<GpuProverSetupData<CompressionProofsTreeHasher>>,
    worker: &Worker,
) -> (ZkSyncCompressionProof, ZkSyncCompressionVerificationKey) {
    let proof_config = circuit.proof_config_for_compression_step();
    let verifier_builder = circuit.into_dyn_verifier_builder();
    let verifier = verifier_builder.create_verifier();
    let gpu_proof_config = GpuProofConfig::from_compression_layer_circuit(&circuit);

    let (proof, vk, is_proof_valid) = match circuit {
        ZkSyncCompressionLayerCircuit::CompressionMode1Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode2Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode3Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode4Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode5Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
    };
    if !is_proof_valid {
        panic!("Proof is invalid");
    }

    (proof, vk)
}

pub fn prove_compression_wrapper_circuit(
    circuit: ZkSyncCompressionForWrapperCircuit,
    setup_data: &mut Option<GpuProverSetupData<CompressionTreeHasherForWrapper>>,
    worker: &Worker,
) -> (
    ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
) {
    let proof_config = circuit.proof_config_for_compression_step();
    let verifier_builder = circuit.into_dyn_verifier_builder();
    let verifier = verifier_builder.create_verifier();
    let gpu_proof_config = GpuProofConfig::from_compression_wrapper_circuit(&circuit);

    let (proof, vk, is_proof_valid) = match circuit {
        ZkSyncCompressionForWrapperCircuit::CompressionMode1Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode2Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode3Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode4Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode5Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                setup_data,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
    };
    if !is_proof_valid {
        panic!("Proof is invalid");
    }

    (proof, vk)
}

pub fn inner_prove_compression_layer_circuit<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
    proof_cfg: ProofConfig,
    gpu_cfg: GpuProofConfig,
    setup_data: &mut Option<GpuProverSetupData<CompressionProofsTreeHasher>>,
    worker: &Worker,
) -> (ZkSyncCompressionProof, ZkSyncCompressionVerificationKey) {
    let local_setup_data = match setup_data.take() {
        Some(setup_data) => setup_data,
        None => {
            let (setup_cs, finalization_hint) = synthesize_compression_circuit::<
                _,
                SetupCSConfig,
                Global,
            >(circuit.clone(), true, None);
            let (base_setup, _setup, setup_tree, variables_hint, witnesses_hint, vk) = setup_cs.prepare_base_setup_with_precomputations_and_vk::<CompressionProofsTranscript, CompressionProofsTreeHasher>(proof_cfg.clone(), worker);
            let gpu_setup = GpuSetup::from_setup_and_hints(
                base_setup,
                setup_tree,
                variables_hint,
                witnesses_hint,
                worker,
            )
            .unwrap();
            GpuProverSetupData {
                setup: gpu_setup,
                vk,
                finalization_hint: finalization_hint.unwrap(),
            }
        }
    };
    let (proving_cs, _) = synthesize_compression_circuit::<_, ProvingCSConfig, Global>(
        circuit.clone(),
        true,
        Some(&local_setup_data.finalization_hint),
    );
    let witness = proving_cs.witness.as_ref().unwrap();
    let domain_size = local_setup_data.vk.fixed_parameters.domain_size as usize;
    let config = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
    let cache_strategy = CacheStrategy {
        setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        commitment: CommitmentCacheStrategy::CacheCosetCaps,
    };
    let gpu_proof = gpu_prove_from_external_witness_data_with_cache_strategy::<
        CompressionProofsTranscript,
        CompressionProofsTreeHasher,
        CF::ThisLayerPoW,
        Global,
    >(
        &gpu_cfg,
        witness,
        proof_cfg.clone(),
        &local_setup_data.setup,
        &local_setup_data.vk,
        (),
        worker,
        cache_strategy,
    )
    .expect("gpu proof");
    drop(ctx);
    let proof = gpu_proof.into();
    let vk = local_setup_data.vk.clone();
    setup_data.replace(local_setup_data);
    (proof, vk)
}

pub fn inner_prove_compression_wrapper_circuit<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
    proof_cfg: ProofConfig,
    gpu_cfg: GpuProofConfig,
    setup_data: &mut Option<GpuProverSetupData<CompressionTreeHasherForWrapper>>,
    worker: &Worker,
) -> (
    ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
) {
    let local_setup_data = match setup_data.take() {
        Some(setup_data) => setup_data,
        None => {
            let (setup_cs, finalization_hint) = synthesize_compression_circuit::<
                _,
                SetupCSConfig,
                Global,
            >(circuit.clone(), true, None);
            let (base_setup, _setup, setup_tree, variables_hint, witnesses_hint, vk) = setup_cs.prepare_base_setup_with_precomputations_and_vk::<CompressionTranscriptForWrapper, CompressionTreeHasherForWrapper>(proof_cfg.clone(), worker);
            let gpu_setup = GpuSetup::from_setup_and_hints(
                base_setup,
                setup_tree,
                variables_hint,
                witnesses_hint,
                worker,
            )
            .unwrap();
            GpuProverSetupData {
                setup: gpu_setup,
                vk,
                finalization_hint: finalization_hint.unwrap(),
            }
        }
    };
    let (proving_cs, _) = synthesize_compression_circuit::<_, ProvingCSConfig, Global>(
        circuit,
        true,
        Some(&local_setup_data.finalization_hint),
    );
    let witness = proving_cs.witness.as_ref().unwrap();
    let domain_size = local_setup_data.vk.fixed_parameters.domain_size as usize;
    let config = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
    let cache_strategy = CacheStrategy {
        setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        commitment: CommitmentCacheStrategy::CacheCosetCaps,
    };
    let gpu_proof = gpu_prove_from_external_witness_data_with_cache_strategy::<
        CompressionTranscriptForWrapper,
        CompressionTreeHasherForWrapper,
        CF::ThisLayerPoW,
        Global,
    >(
        &gpu_cfg,
        witness,
        proof_cfg.clone(),
        &local_setup_data.setup,
        &local_setup_data.vk,
        (),
        worker,
        cache_strategy,
    )
    .expect("gpu proof");
    drop(ctx);
    let vk = local_setup_data.vk.clone();
    setup_data.replace(local_setup_data);
    (gpu_proof.into(), vk)
}

pub fn verify_compression_layer_circuit<CF: ProofCompressionFunction>(
    _circuit: CompressionLayerCircuit<CF>,
    proof: &ZkSyncCompressionProof,
    vk: &ZkSyncCompressionVerificationKey,
    verifier: Verifier<F, EXT>,
) -> bool {
    verifier.verify::<CompressionProofsTreeHasher, CompressionProofsTranscript, CF::ThisLayerPoW>(
        (),
        vk,
        proof,
    )
}

pub fn verify_compression_wrapper_circuit<CF: ProofCompressionFunction>(
    _circuit: CompressionLayerCircuit<CF>,
    proof: &ZkSyncCompressionProofForWrapper,
    vk: &ZkSyncCompressionVerificationKeyForWrapper,
    verifier: Verifier<F, EXT>,
) -> bool {
    verifier.verify::<CompressionProofsTreeHasherForWrapper, CompressionTranscriptForWrapper, CF::ThisLayerPoW>(
        (),
        vk,
        proof,
    )
}
