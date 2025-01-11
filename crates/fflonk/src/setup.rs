use bellman::{
    kate_commitment::{Crs, CrsForMonomialForm},
    plonk::{
        better_better_cs::cs::{
            Assembly, Circuit, GateInternal, MainGate, PlonkConstraintSystemParams, PolyIdentifier,
            Setup, SynthesisMode, SynthesisModeGenerateSetup,
        },
        better_cs::keys::{read_curve_affine, read_fr, write_curve_affine, write_fr_vec},
    },
    worker::Worker,
};
use fflonk_cpu::{FflonkAssembly, FflonkVerificationKey};

use super::*;

use crate::HostAllocator;

pub struct FflonkDeviceSetup<E: Engine, C: Circuit<E>, A: HostAllocator = GlobalHost> {
    pub main_gate_selector_monomials: [Vec<E::Fr, A>; 5],
    pub variable_indexes: [Vec<u32, A>; 3],
    pub c0_commitment: E::G1Affine,
    pub g2_elems: [E::G2Affine; 2],
    _c: std::marker::PhantomData<C>,
}

impl<E, C, A> FflonkDeviceSetup<E, C, A>
where
    E: Engine,
    C: Circuit<E>,
    A: HostAllocator,
{
    fn transform_indexes<P: PlonkConstraintSystemParams<E>, MG: MainGate<E>, S: SynthesisMode>(
        setup_assembly: &Assembly<E, P, MG, S, A>,
    ) -> [Vec<u32, A>; 3] {
        assert!(S::PRODUCE_SETUP);
        assert_eq!(setup_assembly.is_finalized, false);
        let raw_trace_len = setup_assembly.n();
        let num_input_gates = setup_assembly.num_input_gates;
        println!("Transforming indexes into u32 values");
        let mut h_all_transformed_variables = vec![];
        for col_idx in 0..3 {
            let idx = PolyIdentifier::VariablesPolynomial(col_idx);
            let input_rows = setup_assembly.inputs_storage.state_map.get(&idx).unwrap();
            let aux_rows = setup_assembly.aux_storage.state_map.get(&idx).unwrap();
            let mut transformed_variables = Vec::with_capacity_in(raw_trace_len, A::default());
            for var in input_rows.iter().chain(aux_rows.iter()) {
                let new_var = match var.get_unchecked() {
                    bellman::plonk::cs::gates::Index::Input(0) => unreachable!(),
                    bellman::plonk::cs::gates::Index::Aux(0) => 0u32,
                    bellman::plonk::cs::gates::Index::Input(input_idx) => input_idx as u32,
                    bellman::plonk::cs::gates::Index::Aux(aux_idx) => {
                        (num_input_gates + aux_idx) as u32
                    }
                };
                transformed_variables.push(new_var);
            }
            assert_eq!(transformed_variables.len(), raw_trace_len);
            h_all_transformed_variables.push(transformed_variables);
        }

        h_all_transformed_variables.try_into().unwrap()
    }

    pub fn create_setup_on_device(circuit: &C) -> CudaResult<Self> {
        let mut setup_assembly = FflonkAssembly::<E, SynthesisModeGenerateSetup, A>::new();
        circuit.synthesize(&mut setup_assembly).unwrap();
        Self::create_setup_from_assembly_on_device(&setup_assembly)
    }

    pub fn create_setup_from_assembly_on_device<
        P: PlonkConstraintSystemParams<E>,
        MG: MainGate<E>,
        S: SynthesisMode,
    >(
        setup_assembly: &Assembly<E, P, MG, S, A>,
    ) -> CudaResult<Self> {
        assert!(S::PRODUCE_SETUP);
        assert!(setup_assembly.is_finalized == false);
        let raw_trace_len = setup_assembly.n();
        let domain_size = (raw_trace_len + 1).next_power_of_two();
        assert!(setup_assembly.is_satisfied());
        assert_eq!(
            GateInternal::name(&setup_assembly.main_gate),
            GateInternal::name(C::declare_used_gates().unwrap()[0].as_ref()),
        );
        let num_cols = GateInternal::<E>::variable_polynomials(&setup_assembly.main_gate).len();
        assert_eq!(num_cols, 3);

        let h_all_transformed_variables = Self::transform_indexes(&setup_assembly);

        let mut _context = None;
        if is_context_initialized() == false {
            _context = Some(DeviceContextWithSingleDevice::init(domain_size)?)
        }
        assert!(is_context_initialized());
        let stream = bc_stream::new().unwrap();
        // let substream = bc_stream::new().unwrap();
        println!("Loading u32 indexes from host");
        let mut all_transformed_variables =
            DVec::allocate_zeroed_on(num_cols * domain_size, _tmp_mempool(), stream);
        for (src, dst) in h_all_transformed_variables
            .iter()
            .zip(all_transformed_variables.chunks_mut(domain_size))
        {
            let (dst, _) = dst.split_at_mut(raw_trace_len);
            mem::h2d_on(src, dst, stream)?;
        }
        let permutation_monomials =
            materialize_permutation_polys(&all_transformed_variables, domain_size, stream)?;
        let mut setup_polys = setup_assembly.make_setup_polynomials(false).unwrap();
        let mut main_gate_selectors = vec![];
        let mut h_main_gate_selectors = vec![];
        let mut d2h_events = vec![];
        let substream = bc_stream::new().unwrap();
        println!("Reading selector values from Assembly");
        for selector_col in GateInternal::<E>::setup_polynomials(&setup_assembly.main_gate) {
            let raw_main_gate_selectors = setup_polys.remove(selector_col).unwrap();
            assert_eq!(raw_main_gate_selectors.size(), raw_trace_len);
            let mut values = Poly::zero(domain_size);
            let event = bc_event::new().unwrap();
            mem::h2d_on(
                raw_main_gate_selectors.as_ref(),
                &mut values.as_mut()[..raw_trace_len],
                stream,
            )?;
            event.record(substream).unwrap();
            stream.wait(event).unwrap();
            let monomial = values.ifft_on(stream)?;
            let event = bc_event::new().unwrap();
            let h_monomial = monomial.as_ref().to_vec_in(stream, A::default()).unwrap();
            event.record(substream).unwrap();
            d2h_events.push(event);
            h_main_gate_selectors.push(h_monomial);
            main_gate_selectors.push(monomial);
        }
        assert_eq!(main_gate_selectors.len(), 5);
        assert_eq!(h_main_gate_selectors.len(), 5);

        println!("Computing combined monomial on the device");
        let mut combined_monomial = Poly::zero(MAX_COMBINED_DEGREE_FACTOR * domain_size);
        combine_monomials(
            main_gate_selectors
                .iter()
                .chain(permutation_monomials.iter()),
            &mut combined_monomial,
            8,
            stream,
        )?;
        println!("Computing preprocessing combined commitment on the device");
        let c0_commitment = msm::<E>(combined_monomial.as_ref(), domain_size, stream)?;
        let g2_elems = get_g2_elems_from_compact_crs::<E>();
        d2h_events.into_iter().for_each(|e| e.sync().unwrap());
        stream.sync().unwrap();
        Ok(Self {
            variable_indexes: h_all_transformed_variables.try_into().unwrap(),
            main_gate_selector_monomials: h_main_gate_selectors.try_into().unwrap(),
            c0_commitment,
            g2_elems,
            _c: std::marker::PhantomData,
        })
    }

    pub fn create_setup_on_host(circuit: &C, worker: &Worker) -> Self {
        let mut setup_assembly = FflonkAssembly::<E, SynthesisModeGenerateSetup, A>::new();
        circuit.synthesize(&mut setup_assembly).unwrap();
        Self::create_setup_from_assembly_on_host(setup_assembly, worker)
    }

    pub fn create_setup_from_assembly_on_host<P, MG, S>(
        mut setup_assembly: Assembly<E, P, MG, S, A>,
        worker: &Worker,
    ) -> Self
    where
        P: PlonkConstraintSystemParams<E>,
        MG: MainGate<E>,
        S: SynthesisMode,
    {
        assert!(S::PRODUCE_SETUP);
        assert_eq!(setup_assembly.is_finalized, false);
        assert!(setup_assembly.is_satisfied());
        let raw_trace_len = setup_assembly.n();
        let domain_size = (raw_trace_len + 1).next_power_of_two();
        let h_all_transformed_variables = Self::transform_indexes(&setup_assembly);

        let num_cols = GateInternal::<E>::variable_polynomials(&setup_assembly.main_gate).len();
        assert_eq!(num_cols, 3);
        assert_eq!(
            GateInternal::name(&setup_assembly.main_gate),
            GateInternal::name(C::declare_used_gates().unwrap()[0].as_ref()),
        );
        setup_assembly.finalize();
        let setup: Setup<E, C> = setup_assembly.create_setup(worker).unwrap();
        assert_eq!(setup.n + 1, domain_size);

        // comitment to the combined polynomial
        let c0_monomial = fflonk::compute_combined_setup_monomial(&setup, domain_size).unwrap();
        let mon_crs = unsafe {
            let crs = init_crs(&worker, domain_size);
            let g1_bases = std::sync::Arc::try_unwrap(crs.g1_bases).unwrap();
            let g2_bases = std::sync::Arc::try_unwrap(crs.g2_monomial_bases).unwrap();
            let transmuted_g1: Vec<E::G1Affine> = std::mem::transmute(g1_bases);
            let transmuted_g2: Vec<E::G2Affine> = std::mem::transmute(g2_bases);
            Crs::<E, CrsForMonomialForm>::new(transmuted_g1, transmuted_g2)
        };
        let c0_commitment =
            bellman::kate_commitment::commit_using_monomials(&c0_monomial, &mon_crs, worker)
                .unwrap();

        let mut main_gate_selectors = vec![];
        for col in setup.gate_setup_monomials.into_iter() {
            let mut new = Vec::with_capacity_in(col.size(), A::default());
            new.extend(col.into_coeffs());
            main_gate_selectors.push(new);
        }

        let g2_elems = [mon_crs.g2_monomial_bases[0], mon_crs.g2_monomial_bases[1]];

        Self {
            variable_indexes: h_all_transformed_variables.try_into().unwrap(),
            main_gate_selector_monomials: main_gate_selectors.try_into().unwrap(),
            c0_commitment,
            g2_elems,
            _c: std::marker::PhantomData,
        }
    }

    pub fn get_verification_key(&self) -> FflonkVerificationKey<E, C> {
        let raw_trace_len = self.variable_indexes[0].len();
        let domain_size = self.main_gate_selector_monomials[0].len();
        assert_eq!((raw_trace_len + 1).next_power_of_two(), domain_size);
        let n = domain_size - 1;
        let num_inputs = 1;
        let num_state_polys = 3;
        let num_witness_polys = 0;
        let total_lookup_entries_length = 0;
        FflonkVerificationKey::new(
            n,
            self.c0_commitment,
            num_inputs,
            num_state_polys,
            num_witness_polys,
            total_lookup_entries_length,
            self.g2_elems,
        )
    }
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
        let mut variable_indexes = vec![];
        for _ in 0..num_polys {
            let num_values = reader.read_u64::<BigEndian>()?;
            let mut indexes = Vec::with_capacity_in(num_values as usize, A::default());
            for _ in 0..num_values {
                let el = reader.read_u32::<BigEndian>()?;
                indexes.push(el);
            }
            variable_indexes.push(indexes);
        }

        let c0_commitment = read_curve_affine(&mut reader)?;
        let g2_first = read_curve_affine(&mut reader)?;
        let g2_second = read_curve_affine(&mut reader)?;

        Ok(Self {
            main_gate_selector_monomials: main_gate_selector_monomials.try_into().unwrap(),
            variable_indexes: variable_indexes.try_into().unwrap(),
            c0_commitment,
            g2_elems: [g2_first, g2_second],
            _c: std::marker::PhantomData,
        })
    }

    pub fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        use byteorder::{BigEndian, WriteBytesExt};
        writer.write_u64::<BigEndian>(self.main_gate_selector_monomials.len() as u64)?;
        for mon in self.main_gate_selector_monomials.iter() {
            write_fr_vec(&mon, &mut writer)?;
        }
        writer.write_u64::<BigEndian>(self.variable_indexes.len() as u64)?;
        for col in self.variable_indexes.iter() {
            writer.write_u64::<BigEndian>(col.len() as u64)?;
            for el in col {
                writer.write_u32::<BigEndian>(*el)?;
            }
        }
        write_curve_affine(&self.c0_commitment, &mut writer)?;
        write_curve_affine(&self.g2_elems[0], &mut writer)?;
        write_curve_affine(&self.g2_elems[1], writer)
    }
}

pub fn read_raw_fr_vec<F: PrimeField, R: std::io::Read, A: Allocator + Default>(
    mut src: R,
) -> std::io::Result<Vec<F, A>> {
    use byteorder::{BigEndian, ReadBytesExt};
    let num_values = src.read_u32::<BigEndian>()? as usize;
    let mut values = Vec::with_capacity_in(num_values, A::default());
    unsafe {
        values.set_len(num_values);
        let buf = std::slice::from_raw_parts_mut(
            values.as_mut_ptr() as *mut u8,
            num_values * std::mem::size_of::<F>(),
        );
        // src.read_exact(buf)?;
        let mut dst = &mut buf[..];
        std::io::copy(&mut src, &mut dst)?;
    }

    Ok(values)
}

pub fn write_raw_fr_slice<F: PrimeField, W: std::io::Write>(
    src_values: &[F],
    mut dst: W,
) -> std::io::Result<()> {
    use byteorder::{BigEndian, WriteBytesExt};
    let num_values = src_values.len();
    assert!(num_values < u32::MAX as usize);
    dst.write_u32::<BigEndian>(num_values as u32)?;
    unsafe {
        let buf = std::slice::from_raw_parts_mut(
            src_values.as_ptr() as *mut u8,
            num_values * std::mem::size_of::<F>(),
        );
        let mut reader = &buf[..];
        // dst.write_all(buf)?;
        std::io::copy(&mut reader, &mut dst)?;
    }

    Ok(())
}
