use std::alloc::Global;

use super::*;
use bellman::{
    bn256::Bn256,
    kate_commitment::{Crs, CrsForMonomialForm},
    plonk::{
        better_better_cs::cs::{
            Circuit, GateInternal, Setup, SynthesisMode, SynthesisModeGenerateSetup,
        },
        better_cs::keys::{read_curve_affine, read_fr, write_curve_affine, write_fr_vec},
        cs::variable::{Index, Variable},
    },
    worker::Worker,
};
use fflonk::{FflonkAssembly, FflonkSnarkVerifierCircuit};
use fflonk_cpu::FflonkVerificationKey;

pub type FflonkSnarkVerifierCircuitDeviceSetup =
    FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, Global>;

pub struct FflonkDeviceSetup<E: Engine, C: Circuit<E>, A: HostAllocator = GlobalHost> {
    // We could use bit representation but elems in lagrange basis are also fine
    pub main_gate_selector_monomials: [Vec<E::Fr, A>; 5],
    pub permutation_monomials: [Vec<E::Fr, A>; 3],
    // Transform 64-bit variable indexes into 32-bits
    // pub variable_indexes: [Vec<u32, A>; 3],
    pub c0_commitment: E::G1Affine,
    pub g2_elems: [E::G2Affine; 2],
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

        let c0_commitment = read_curve_affine(&mut reader)?;
        let g2_first = read_curve_affine(&mut reader)?;
        let g2_second = read_curve_affine(&mut reader)?;

        Ok(Self {
            main_gate_selector_monomials: main_gate_selector_monomials.try_into().unwrap(),
            permutation_monomials: permutation_monomials.try_into().unwrap(),
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
        writer.write_u64::<BigEndian>(self.permutation_monomials.len() as u64)?;
        for mon in self.permutation_monomials.iter() {
            write_fr_vec(&mon, &mut writer)?;
        }
        write_curve_affine(&self.c0_commitment, &mut writer)?;
        write_curve_affine(&self.g2_elems[0], &mut writer)?;
        write_curve_affine(&self.g2_elems[1], writer)
    }
}

impl<E: Engine, C: Circuit<E>, A: HostAllocator> FflonkDeviceSetup<E, C, A> {
    pub fn create_setup_on_host(circuit: &C, worker: &Worker) -> FflonkDeviceSetup<E, C, A> {
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
        let mut permutations = vec![];
        for col in setup.permutation_monomials.into_iter() {
            let mut new = Vec::with_capacity_in(col.size(), A::default());
            new.extend(col.into_coeffs());
            permutations.push(new);
        }

        let g2_elems = [mon_crs.g2_monomial_bases[0], mon_crs.g2_monomial_bases[1]];
        Self {
            main_gate_selector_monomials: main_gate_selectors.try_into().unwrap(),
            permutation_monomials: permutations.try_into().unwrap(),
            c0_commitment,
            g2_elems,
            _c: std::marker::PhantomData,
        }
    }

    pub fn create_setup_on_device(
        circuit: &C,
        worker: &Worker,
    ) -> CudaResult<FflonkDeviceSetup<E, C, A>> {
        let mut setup_assembly = FflonkAssembly::<E, SynthesisModeGenerateSetup>::new();
        circuit
            .synthesize(&mut setup_assembly)
            .expect("synthesize circuit");
        assert!(setup_assembly.num_inputs >= 1);
        assert!(setup_assembly.inputs_storage.setup_map.len() >= 1);
        setup_assembly.finalize();
        let domain_size = setup_assembly.n() + 1;
        assert!(domain_size.is_power_of_two());

        Self::create_setup_from_assembly_on_device(&setup_assembly, worker)
    }

    pub fn get_verification_key(&self) -> FflonkVerificationKey<E, C> {
        let n = (1 << L1_VERIFIER_DOMAIN_SIZE_LOG) - 1;
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

    pub fn create_setup_from_assembly_on_device<S: SynthesisMode>(
        assembly: &FflonkAssembly<E, S>,
        worker: &Worker,
    ) -> CudaResult<FflonkDeviceSetup<E, C, A>> {
        assert!(assembly.is_finalized);
        assert!(S::PRODUCE_SETUP, "Assembly should hold setup values");

        let domain_size = assembly.n() + 1;
        assert!(domain_size.is_power_of_two());
        let mut setup_polys = assembly.make_setup_polynomials(false).unwrap();
        let mut main_gate_selectors = vec![];
        for col in GateInternal::<E>::setup_polynomials(&assembly.main_gate) {
            let unpadded_main_gate_selectors = setup_polys.remove(col).unwrap();
            main_gate_selectors.push(unpadded_main_gate_selectors);
        }
        assert_eq!(main_gate_selectors.len(), 5);
        let permutation_polys = assembly.make_permutations(worker).unwrap();
        assert_eq!(permutation_polys.len(), 3);

        let mut _context = None;
        if is_context_initialized() == false {
            _context = Some(DeviceContextWithSingleDevice::init(domain_size)?)
        }
        assert!(is_context_initialized());

        // move to device and compute values there
        println!("Reading selector values from Assembly");
        let stream = bc_stream::new().unwrap();
        let substream = bc_stream::new().unwrap();
        let mut selector_monomials = vec![];
        for col in main_gate_selectors {
            assert_eq!(col.size() + 1, domain_size);
            let event = bc_event::new().unwrap();
            let mut buf = DVec::allocate_zeroed(domain_size);
            mem::h2d_on(col.as_ref(), &mut buf[..domain_size - 1], substream)?;
            let values = Poly::from_buffer(buf);
            event.sync().unwrap();
            let monomial = values.ifft_on(stream)?;
            selector_monomials.push(monomial);
        }
        println!("Reading permutation values from Assembly");
        let mut permutation_monomials = vec![];
        for col in permutation_polys {
            assert_eq!(col.size(), domain_size);
            let event = bc_event::new().unwrap();
            let mut buf = DVec::allocate_zeroed(domain_size);
            mem::h2d_on(col.as_ref(), &mut buf, substream)?;
            let values = Poly::from_buffer(buf);
            event.sync().unwrap();
            let monomial = values.ifft_on(stream)?;
            permutation_monomials.push(monomial);
        }

        println!("Computing combined monomial on the device");
        let mut combined_monomial = Poly::zero(9 * domain_size);
        combine_monomials(
            selector_monomials
                .iter()
                .chain(permutation_monomials.iter()),
            &mut combined_monomial,
            8,
            stream,
        )?;

        println!("Computing preprocessing combined commitment on the device");
        let c0_commitment = msm::<E>(combined_monomial.as_ref(), domain_size, stream)?;

        println!("Moving selector monomials back to the Host");
        let mut main_gate_selectors = vec![];
        for col in selector_monomials {
            let mut h_monomial = Vec::with_capacity_in(domain_size, A::default());
            unsafe {
                h_monomial.set_len(domain_size);
            }
            mem::d2h_on(col.as_ref(), &mut h_monomial, stream).unwrap();
            main_gate_selectors.push(h_monomial);
        }
        println!("Moving permutation monomials back to the Host");
        let mut permutations = vec![];
        for col in permutation_monomials {
            let mut h_monomial = Vec::with_capacity_in(domain_size, A::default());
            unsafe {
                h_monomial.set_len(domain_size);
            }
            mem::d2h_on(col.as_ref(), &mut h_monomial, stream).unwrap();
            permutations.push(h_monomial);
        }
        stream.sync().unwrap();
        println!("Reading G2 elems from CRS file");
        let g2_elems = get_g2_elems_from_compact_crs::<E>();
        Ok(FflonkDeviceSetup {
            main_gate_selector_monomials: main_gate_selectors.try_into().unwrap(),
            permutation_monomials: permutations.try_into().unwrap(),
            c0_commitment,
            g2_elems,
            _c: std::marker::PhantomData,
        })
    }
}

unsafe fn _transform_variables(
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
