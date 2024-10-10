use snark_wrapper::franklin_crypto::bellman::bn256::{Bn256, Fr};
use snark_wrapper::franklin_crypto::bellman::PrimeField;
use snark_wrapper::rescue_poseidon::poseidon2::Poseidon2Params;
use snark_wrapper::rescue_poseidon::HashParams;

const TEMPLATE_PATH: &str = "native/poseidon2/bn/poseidon2_bn_constants.cuh.template";
const RESULT_PATH: &str = "generated/poseidon2_bn_constants.cuh";

const RATE: usize = 2;
const CAPACITY: usize = 1;
const WIDTH: usize = RATE + CAPACITY;
const CHUNK_BY: usize = 3;

type Params = Poseidon2Params<Bn256, RATE, WIDTH>;

fn split_u64(value: u64) -> (u32, u32) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    (lo, hi)
}

fn get_field_string(value: &Fr) -> String {
    let mut result = String::new();
    for x in value.into_raw_repr().0 {
        let (lo, hi) = split_u64(x);
        result.push_str(format!("{lo:#010x},{hi:#010x},").as_str());
    }
    result
}

fn get_field_array_string(values: &[Fr]) -> String {
    let mut result = String::new();
    for x in values {
        result.push('{');
        result.push_str(get_field_string(x).as_str());
        result.push_str("},");
    }
    result
}

fn get_field_2d_array_string<const COUNT: usize>(values: &[&[Fr; COUNT]]) -> String {
    let mut result = String::from("{\\\n");
    for &row in values {
        result.push_str("  {");
        result.push_str(get_field_array_string(row).as_str());
        result.push_str("},\\\n");
    }
    result.push('}');
    result
}

fn get_round_constants(params: &Params) -> String {
    let num_full_rounds = params.number_of_full_rounds();
    let num_partial_rounds = params.number_of_partial_rounds();
    let chunks: Vec<&[Fr; 3]> = (0..num_full_rounds + num_partial_rounds)
        .map(|round| params.constants_of_round(round))
        .collect();
    get_field_2d_array_string(&chunks)
}

pub(super) fn generate() {
    let params = Params::default();
    let num_full_rounds = params.number_of_full_rounds();
    assert_eq!(num_full_rounds & 1, 0);
    let half_num_full_rounds = num_full_rounds >> 1;
    let num_partial_rounds = params.number_of_partial_rounds();
    let replacements = [
        ("RATE", RATE.to_string()),
        ("CAPACITY", CAPACITY.to_string()),
        ("CHUNK_BY", CHUNK_BY.to_string()),
        ("HALF_NUM_FULL_ROUNDS", half_num_full_rounds.to_string()),
        ("NUM_PARTIAL_ROUNDS", num_partial_rounds.to_string()),
        ("ROUND_CONSTANTS_VALUES", get_round_constants(&params)),
    ];
    super::template::generate(&replacements, TEMPLATE_PATH, RESULT_PATH);
}
