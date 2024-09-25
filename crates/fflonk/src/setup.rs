use std::alloc::Global;

use super::*;
use bellman::{
    bn256::Bn256,
    kate_commitment::{Crs, CrsForMonomialForm},
    plonk::{
        better_better_cs::cs::{Circuit, PolyIdentifier, Setup, SynthesisModeGenerateSetup},
        better_cs::keys::{
            read_curve_affine, read_fr, read_fr_vec, write_curve_affine, write_fr_vec,
        },
        cs::variable::{Index, Variable},
    },
    worker::Worker,
};
use fflonk::{FflonkAssembly, FflonkSetup, FflonkSnarkVerifierCircuit};

pub type FflonkSnarkVerifierCircuitDeviceSetup =
    FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, Global>;

pub struct FflonkDeviceSetup<E: Engine, C: Circuit<E>, A: HostAllocator = Global> {
    // We could use bit representation but elems in lagrange basis are also fine
    pub main_gate_selector_monomials: [Vec<E::Fr, A>; 5],
    pub permutation_monomials: [Vec<E::Fr, A>; 3],
    // Transform 64-bit variable indexes into 32-bits
    // pub variable_indexes: [Vec<u32, A>; 3],
    pub c0_commitment: E::G1Affine,
    _c: std::marker::PhantomData<C>,
}

impl<E: Engine, C: Circuit<E>, A: HostAllocator> FflonkDeviceSetup<E, C, A> {
    pub fn read<R: std::io::Read>(mut reader: R) -> std::io::Result<Self> {
        use byteorder::BigEndian;
        use byteorder::ReadBytesExt;

        let num_polys = reader.read_u64::<BigEndian>()?;
        assert_eq!(num_polys, 5);
        let mut main_gate_selector_monomials = vec![];
        for _ in 0..num_polys {
            let num_values = reader.read_u64::<BigEndian>()?;
            let mut coeffs = Vec::with_capacity_in(num_values as usize, A::default());
            for _ in 0..num_values {
                let el = read_fr(&mut reader)?;
                coeffs.push(el);
            }
            main_gate_selector_monomials.push(coeffs);
        }

        let num_polys = reader.read_u64::<BigEndian>()?;
        assert_eq!(num_polys, 3);
        let mut permutation_monomials = vec![];
        for _ in 0..num_polys {
            let num_values = reader.read_u64::<BigEndian>()?;
            let mut coeffs = Vec::with_capacity_in(num_values as usize, A::default());
            for _ in 0..num_values {
                let el = read_fr(&mut reader)?;
                coeffs.push(el);
            }
            permutation_monomials.push(coeffs);
        }

        let c0_commitment = read_curve_affine(reader)?;

        Ok(Self {
            main_gate_selector_monomials: main_gate_selector_monomials.try_into().unwrap(),
            permutation_monomials: permutation_monomials.try_into().unwrap(),
            c0_commitment,
            _c: std::marker::PhantomData,
        })
    }

    pub fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        use byteorder::{BigEndian, WriteBytesExt};
        writer.write_u64::<BigEndian>(self.main_gate_selector_monomials.len() as u64)?;
        for mon in self.main_gate_selector_monomials.iter() {
            write_fr_vec(&mon, &mut writer)?;
        }
        writer.write_u64::<BigEndian>(self.permutation_monomials.len() as u64)?;
        for mon in self.permutation_monomials.iter() {
            write_fr_vec(&mon, &mut writer)?;
        }
        write_curve_affine(&self.c0_commitment, writer)
    }
}

impl<E: Engine, C: Circuit<E>> FflonkDeviceSetup<E, C> {
    pub fn create_setup_on_host(
        circuit: &C,
        mon_crs: &Crs<E, CrsForMonomialForm>,
        worker: &Worker,
    ) -> FflonkDeviceSetup<E, C> {
        let mut setup_assembly = FflonkAssembly::<E, SynthesisModeGenerateSetup>::new();
        circuit
            .synthesize(&mut setup_assembly)
            .expect("synthesize circuit");
        assert!(setup_assembly.num_inputs >= 1);
        assert!(setup_assembly.inputs_storage.setup_map.len() >= 1);

        // We want to keep only raw(unpadded) trace column indexes as
        // permutation polynomials will be constructed during proof generation

        setup_assembly.finalize();

        let setup: Setup<E, C> = setup_assembly.create_setup(worker).unwrap();
        let domain_size = setup.n + 1;
        assert!(domain_size.is_power_of_two());

        // comitment to the combined polynomial
        let c0_monomial = fflonk::compute_combined_setup_monomial(&setup, domain_size).unwrap();
        let c0_commitment =
            bellman::kate_commitment::commit_using_monomials(&c0_monomial, &mon_crs, worker)
                .unwrap();
        let host_setup = FflonkSetup {
            original_setup: setup,
            c0_commitment,
        };
        Self::from_host_setup(host_setup)
    }

    pub fn from_host_setup(host_setup: FflonkSetup<E, C>) -> Self {
        let c0_commitment = host_setup.c0_commitment;
        let Setup {
            gate_setup_monomials,
            permutation_monomials,
            ..
        } = host_setup.original_setup;

        let main_gate_selector_monomials = gate_setup_monomials
            .into_iter()
            .map(|col| col.into_coeffs())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let permutation_monomials = permutation_monomials
            .into_iter()
            .map(|col| col.into_coeffs())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            main_gate_selector_monomials,
            permutation_monomials,
            c0_commitment,
            _c: std::marker::PhantomData,
        }
    }
}

unsafe fn transform_variables(
    variables: &[Variable],
    result: &mut [u32],
    num_aux_variables: usize,
    worker: &Worker,
) {
    assert_eq!(variables.len(), result.len());
    worker.scope(variables.len(), |scope, chunk_size| {
        for (src, dst) in variables
            .chunks(chunk_size)
            .zip(result.chunks_mut(chunk_size))
        {
            scope.spawn(move |_| {
                for (a, b) in src.iter().zip(dst.iter_mut()) {
                    *b = match a.get_unchecked() {
                        Index::Input(0) => unreachable!(),
                        Index::Input(idx) => (num_aux_variables + idx) as u32,
                        Index::Aux(0) => 0,
                        Index::Aux(idx) => idx as u32,
                    };
                }
            });
        }
    });
}

pub fn create_setup_on_host<E: Engine, C: Circuit<E>, A: HostAllocator>(
    circuit: &C,
    mon_crs: &Crs<E, CrsForMonomialForm>,
    worker: &Worker,
) -> FflonkDeviceSetup<E, C> {
    let mut setup_assembly = FflonkAssembly::<E, SynthesisModeGenerateSetup>::new();
    circuit
        .synthesize(&mut setup_assembly)
        .expect("synthesize circuit");
    assert!(setup_assembly.num_inputs >= 1);
    assert!(setup_assembly.inputs_storage.setup_map.len() >= 1);

    // We want to keep only raw(unpadded) trace column indexes as
    // permutation polynomials will be constructed during proof generation
    let raw_trace_len = setup_assembly.n();

    setup_assembly.finalize();

    let setup: Setup<E, C> = setup_assembly.create_setup(worker).unwrap();
    let domain_size = setup.n + 1;
    assert!(domain_size.is_power_of_two());

    // comitment to the combined polynomial
    let c0_monomial = fflonk::compute_combined_setup_monomial(&setup, domain_size).unwrap();
    let c0_commitment =
        bellman::kate_commitment::commit_using_monomials(&c0_monomial, &mon_crs, worker).unwrap();

    // now construct selector monomials
    let Setup::<E, C> {
        num_inputs,
        gate_setup_monomials,
        ..
    } = setup;

    let main_gate_selector_monomials: [Vec<E::Fr, A>; 5] = std::array::from_fn(|idx| {
        let mut buf = Vec::with_capacity_in(raw_trace_len, A::default());
        unsafe { buf.set_len(raw_trace_len) };
        buf.copy_from_slice(&gate_setup_monomials[idx].as_ref()[..raw_trace_len]);

        buf
    });

    assert_eq!(num_inputs, setup_assembly.num_inputs);
    let FflonkAssembly {
        inputs_storage,
        aux_storage,
        num_aux: num_aux_variables,
        num_input_gates,
        ..
    } = setup_assembly;
    assert_eq!(inputs_storage.setup_map.len(), 5);
    assert_eq!(aux_storage.setup_map.len(), 5);

    // Then construct unpadded 32bit column indexes from Variables
    let mut column_indexes: [Vec<u32, A>; 3] = std::array::from_fn(|_| {
        let mut buf = Vec::with_capacity_in(raw_trace_len, A::default());
        unsafe { buf.set_len(raw_trace_len) };

        buf
    });

    for row_idx in 0..raw_trace_len {
        let (storage, current_row_idx) = if row_idx < num_input_gates {
            (&inputs_storage, row_idx)
        } else {
            (&aux_storage, row_idx - 1)
        };
        for col_idx in 0..3 {
            column_indexes[col_idx][row_idx] = match storage
                .state_map
                .get(&PolyIdentifier::VariablesPolynomial(col_idx))
                .unwrap()
                .get(current_row_idx)
                .unwrap_or(&Variable::new_unchecked(Index::Aux(0)))
                .get_unchecked()
            {
                bellman::plonk::cs::gates::Index::Input(0) => unreachable!(),
                bellman::plonk::cs::gates::Index::Input(input_idx) => {
                    (num_aux_variables + input_idx) as u32 - 1
                }
                bellman::plonk::cs::gates::Index::Aux(0) => 0,
                bellman::plonk::cs::gates::Index::Aux(aux_idx) => aux_idx as u32,
            }
        }
    }

    // Self {
    //     main_gate_selector_monomials,
    //     variable_indexes: column_indexes,
    //     c0_commitment,
    //     _c: std::marker::PhantomData,
    // }

    todo!();
}
