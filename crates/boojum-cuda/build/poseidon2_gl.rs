use boojum::field::goldilocks::GoldilocksField;
use boojum::field::U64Representable;
use boojum::implementations::poseidon_goldilocks_params::*;

const TEMPLATE_PATH: &str = "native/poseidon2/gl/poseidon2_gl_constants.cuh.template";
const RESULT_PATH: &str = "generated/poseidon2_gl_constants.cuh";

fn split_u64(value: u64) -> (u32, u32) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    (lo, hi)
}

fn get_field_array_string(values: &[GoldilocksField]) -> String {
    let mut result = String::new();
    for x in values {
        let (lo, hi) = split_u64(x.as_u64());
        result.push_str(format!("{{{lo:#010x},{hi:#010x}}},").as_str());
    }
    result
}

fn get_field_2d_array_string<const COUNT: usize>(values: &[[GoldilocksField; COUNT]]) -> String {
    let mut result = String::from('\n');
    for row in values {
        result.push_str("  {");
        result.push_str(get_field_array_string(row).as_str());
        result.push_str("},\n");
    }
    result
}

fn get_all_round_constants() -> String {
    let values = ALL_ROUND_CONSTANTS_AS_FIELD_ELEMENTS;
    assert_eq!(values.len(), STATE_WIDTH * TOTAL_NUM_ROUNDS);
    let chunks: Vec<[GoldilocksField; STATE_WIDTH]> = values
        .chunks(STATE_WIDTH)
        .map(|c| c.try_into().unwrap())
        .collect();
    get_field_2d_array_string(&chunks)
}

pub(super) fn generate() {
    let replacements = [
        ("RATE", RATE.to_string()),
        ("CAPACITY", CAPACITY.to_string()),
        ("HALF_NUM_FULL_ROUNDS", HALF_NUM_FULL_ROUNDS.to_string()),
        ("NUM_PARTIAL_ROUNDS", NUM_PARTIAL_ROUNDS.to_string()),
        ("ALL_ROUND_CONSTANTS", get_all_round_constants()),
    ];
    super::template::generate(&replacements, TEMPLATE_PATH, RESULT_PATH);
}
