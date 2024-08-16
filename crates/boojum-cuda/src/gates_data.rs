use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    static ref HASH_MAP: HashMap<&'static str, u32> = [
        ("boolean_constrait_evaluator", 0),
        ("conditional_swap_gate_constraint_evaluator_1", 1),
        ("conditional_swap_gate_constraint_evaluator_4", 2),
        ("constant_allocator_constraint_evaluator", 3),
        ("dot_product_constraint_evaluator_4", 4),
        ("fma_gate_in_base_without_constant_constraint_evaluator", 5),
        ("fma_gate_in_extension_without_constant_constraint_evaluator_goldilocks_field_goldilocks_ext_2", 6),
        ("matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_external_matrix", 7),
        ("matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_inner_matrix", 8),
        ("parallel_selection_gate_constraint_evaluator_4", 9),
        ("poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_1451e131831047e6", 10),
        ("poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_0ebd1f683a24cfac", 11),
        ("quadratic_combination_constraint_evaluator_4", 12),
        ("reduction_by_powers_gate_constraint_evaluator_4", 13),
        ("reduction_gate_constraint_evaluator_4", 14),
        ("selection_gate_constraint_evaluator", 15),
        ("simple_nonlinearity_gate_constraint_evaluator_7", 16),
        ("u_32_add_constraint_evaluator", 17),
        ("u_8_x_4_constraint_evaluator", 18),
        ("u_32_sub_constraint_evaluator", 19),
        ("u_32_tri_add_carry_as_chunk_constraint_evaluator", 20),
        ("u_int_x_add_constraint_evaluator", 21),
        ("zero_check_evaluator_68a914128e01e473", 22),
        ("zero_check_evaluator_44bc103b1f8540ed", 23),
    ]
    .iter()
    .copied()
    .collect();
}

gate_eval_kernel!(evaluate_boolean_constrait_evaluator_kernel);
gate_eval_kernel!(evaluate_conditional_swap_gate_constraint_evaluator_1_kernel);
gate_eval_kernel!(evaluate_conditional_swap_gate_constraint_evaluator_4_kernel);
gate_eval_kernel!(evaluate_constant_allocator_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_dot_product_constraint_evaluator_4_kernel);
gate_eval_kernel!(evaluate_fma_gate_in_base_without_constant_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_fma_gate_in_extension_without_constant_constraint_evaluator_goldilocks_field_goldilocks_ext_2_kernel);
gate_eval_kernel!(evaluate_matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_external_matrix_kernel);
gate_eval_kernel!(evaluate_matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_inner_matrix_kernel);
gate_eval_kernel!(evaluate_parallel_selection_gate_constraint_evaluator_4_kernel);
gate_eval_kernel!(evaluate_poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_1451e131831047e6_kernel);
gate_eval_kernel!(evaluate_poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_0ebd1f683a24cfac_kernel);
gate_eval_kernel!(evaluate_quadratic_combination_constraint_evaluator_4_kernel);
gate_eval_kernel!(evaluate_reduction_by_powers_gate_constraint_evaluator_4_kernel);
gate_eval_kernel!(evaluate_reduction_gate_constraint_evaluator_4_kernel);
gate_eval_kernel!(evaluate_selection_gate_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_simple_nonlinearity_gate_constraint_evaluator_7_kernel);
gate_eval_kernel!(evaluate_u_32_add_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_u_8_x_4_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_u_32_sub_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_u_32_tri_add_carry_as_chunk_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_u_int_x_add_constraint_evaluator_kernel);
gate_eval_kernel!(evaluate_zero_check_evaluator_68a914128e01e473_kernel);
gate_eval_kernel!(evaluate_zero_check_evaluator_44bc103b1f8540ed_kernel);

fn get_gate_data(id: u32) -> GateData {
    match id {
        0 => GateData {
            name: "boolean_constrait_evaluator",
            contributions_count: 1,
            max_variable_index: Some(0),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_boolean_constrait_evaluator_kernel,
        },
        1 => GateData {
            name: "conditional_swap_gate_constraint_evaluator_1",
            contributions_count: 2,
            max_variable_index: Some(4),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_conditional_swap_gate_constraint_evaluator_1_kernel,
        },
        2 => GateData {
            name: "conditional_swap_gate_constraint_evaluator_4",
            contributions_count: 8,
            max_variable_index: Some(16),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_conditional_swap_gate_constraint_evaluator_4_kernel,
        },
        3 => GateData {
            name: "constant_allocator_constraint_evaluator",
            contributions_count: 1,
            max_variable_index: Some(0),
            max_witness_index: None,
            max_constant_index: Some(0),
            kernel: evaluate_constant_allocator_constraint_evaluator_kernel,
        },
        4 => GateData {
            name: "dot_product_constraint_evaluator_4",
            contributions_count: 1,
            max_variable_index: Some(8),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_dot_product_constraint_evaluator_4_kernel,
        },
        5 => GateData {
            name: "fma_gate_in_base_without_constant_constraint_evaluator",
            contributions_count: 1,
            max_variable_index: Some(3),
            max_witness_index: None,
            max_constant_index: Some(1),
            kernel: evaluate_fma_gate_in_base_without_constant_constraint_evaluator_kernel,
        },
        6 => GateData {
            name: "fma_gate_in_extension_without_constant_constraint_evaluator_goldilocks_field_goldilocks_ext_2",
            contributions_count: 2,
            max_variable_index: Some(7),
            max_witness_index: None,
            max_constant_index: Some(3),
            kernel: evaluate_fma_gate_in_extension_without_constant_constraint_evaluator_goldilocks_field_goldilocks_ext_2_kernel,
        },
        7 => GateData {
            name: "matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_external_matrix",
            contributions_count: 12,
            max_variable_index: Some(23),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_external_matrix_kernel,
        },
        8 => GateData {
            name: "matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_inner_matrix",
            contributions_count: 12,
            max_variable_index: Some(23),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_matrix_multiplication_evaluator_goldilocks_field_12_poseidon_2_goldilocks_inner_matrix_kernel,
        },
        9 => GateData {
            name: "parallel_selection_gate_constraint_evaluator_4",
            contributions_count: 4,
            max_variable_index: Some(12),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_parallel_selection_gate_constraint_evaluator_4_kernel,
        },
        10 => GateData {
            name: "poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_1451e131831047e6",
            contributions_count: 118,
            max_variable_index: Some(129),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_1451e131831047e6_kernel,
        },
        11 => GateData {
            name: "poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_0ebd1f683a24cfac",
            contributions_count: 118,
            max_variable_index: Some(99),
            max_witness_index: Some(29),
            max_constant_index: None,
            kernel: evaluate_poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_0ebd1f683a24cfac_kernel,
        },
        12 => GateData {
            name: "quadratic_combination_constraint_evaluator_4",
            contributions_count: 1,
            max_variable_index: Some(7),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_quadratic_combination_constraint_evaluator_4_kernel,
        },
        13 => GateData {
            name: "reduction_by_powers_gate_constraint_evaluator_4",
            contributions_count: 1,
            max_variable_index: Some(4),
            max_witness_index: None,
            max_constant_index: Some(0),
            kernel: evaluate_reduction_by_powers_gate_constraint_evaluator_4_kernel,
        },
        14 => GateData {
            name: "reduction_gate_constraint_evaluator_4",
            contributions_count: 1,
            max_variable_index: Some(4),
            max_witness_index: None,
            max_constant_index: Some(3),
            kernel: evaluate_reduction_gate_constraint_evaluator_4_kernel,
        },
        15 => GateData {
            name: "selection_gate_constraint_evaluator",
            contributions_count: 1,
            max_variable_index: Some(3),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_selection_gate_constraint_evaluator_kernel,
        },
        16 => GateData {
            name: "simple_nonlinearity_gate_constraint_evaluator_7",
            contributions_count: 1,
            max_variable_index: Some(1),
            max_witness_index: None,
            max_constant_index: Some(0),
            kernel: evaluate_simple_nonlinearity_gate_constraint_evaluator_7_kernel,
        },
        17 => GateData {
            name: "u_32_add_constraint_evaluator",
            contributions_count: 2,
            max_variable_index: Some(4),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_u_32_add_constraint_evaluator_kernel,
        },
        18 => GateData {
            name: "u_8_x_4_constraint_evaluator",
            contributions_count: 2,
            max_variable_index: Some(25),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_u_8_x_4_constraint_evaluator_kernel,
        },
        19 => GateData {
            name: "u_32_sub_constraint_evaluator",
            contributions_count: 2,
            max_variable_index: Some(4),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_u_32_sub_constraint_evaluator_kernel,
        },
        20 => GateData {
            name: "u_32_tri_add_carry_as_chunk_constraint_evaluator",
            contributions_count: 1,
            max_variable_index: Some(16),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_u_32_tri_add_carry_as_chunk_constraint_evaluator_kernel,
        },
        21 => GateData {
            name: "u_int_x_add_constraint_evaluator",
            contributions_count: 2,
            max_variable_index: Some(4),
            max_witness_index: None,
            max_constant_index: Some(0),
            kernel: evaluate_u_int_x_add_constraint_evaluator_kernel,
        },
        22 => GateData {
            name: "zero_check_evaluator_68a914128e01e473",
            contributions_count: 2,
            max_variable_index: Some(2),
            max_witness_index: None,
            max_constant_index: None,
            kernel: evaluate_zero_check_evaluator_68a914128e01e473_kernel,
        },
        23 => GateData {
            name: "zero_check_evaluator_44bc103b1f8540ed",
            contributions_count: 2,
            max_variable_index: Some(1),
            max_witness_index: Some(0),
            max_constant_index: None,
            kernel: evaluate_zero_check_evaluator_44bc103b1f8540ed_kernel,
        },
        _ => panic!("unknown gate id {id}"),
    }
}
