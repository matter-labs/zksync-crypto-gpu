use super::*;
use bellman::{
    kate_commitment::{Crs, CrsForMonomialForm},
    CurveAffine, Engine, Field, PrimeField,
};
use byteorder::{BigEndian, ReadBytesExt};

pub fn create_crs_from_ignition_transcript<S: AsRef<std::ffi::OsStr> + ?Sized>(
    path: &S,
    num_chunks: usize,
) -> Result<
    bellman::kate_commitment::Crs<
        bellman::pairing::bn256::Bn256,
        bellman::kate_commitment::CrsForMonomialForm,
    >,
    bellman::SynthesisError,
> {
    use bellman::pairing::bn256::{Fq, Fq2};
    use std::io::BufRead;
    use std::io::Read;

    let chunk_size = 5_040_000;

    let base_path = std::path::Path::new(&path);

    let mut g1_bases = Vec::with_capacity(100800000 + 1);
    g1_bases.push(<Bn256 as Engine>::G1Affine::one());
    let mut g2_bases = vec![<Bn256 as Engine>::G2Affine::one()];

    for i in 0..num_chunks {
        let full_path = base_path.join(&format!("transcript{:02}.dat", i));
        println!("Opening {}", full_path.to_string_lossy());
        let file =
            std::fs::File::open(full_path).map_err(|e| bellman::SynthesisError::IoError(e))?;
        let mut reader = std::io::BufReader::with_capacity(1 << 24, file);

        // skip 28 bytes
        let mut tmp = [0u8; 28];
        reader.read_exact(&mut tmp).expect("must skip 28 bytes");

        let mut fq_repr = <Fq as PrimeField>::Repr::default();
        let b_coeff = Fq::from_str("3").unwrap();

        fq_repr.as_mut()[0] = 0x3bf938e377b802a8;
        fq_repr.as_mut()[1] = 0x020b1b273633535d;
        fq_repr.as_mut()[2] = 0x26b7edf049755260;
        fq_repr.as_mut()[3] = 0x2514c6324384a86d;

        let c0 = Fq::from_raw_repr(fq_repr).expect("c0 for B coeff for G2");

        fq_repr.as_mut()[0] = 0x38e7ecccd1dcff67;
        fq_repr.as_mut()[1] = 0x65f0b37d93ce0d3e;
        fq_repr.as_mut()[2] = 0xd749d0dd22ac00aa;
        fq_repr.as_mut()[3] = 0x0141b9ce4a688d4d;

        let c1 = Fq::from_raw_repr(fq_repr).expect("c0 for B coeff for G2");

        let b_coeff_fq2 = Fq2 { c0: c0, c1: c1 };

        for _ in 0..chunk_size {
            // we have to manually read X and Y coordinates
            for k in 0..4 {
                fq_repr.as_mut()[k] = reader.read_u64::<BigEndian>().expect("must read u64");
            }

            let x = Fq::from_repr(fq_repr).expect("must be valid field element encoding");

            for k in 0..4 {
                fq_repr.as_mut()[k] = reader.read_u64::<BigEndian>().expect("must read u64");
            }

            let y = Fq::from_repr(fq_repr).expect("must be valid field element encoding");

            // manual on-curve check
            {
                let mut lhs = y;
                lhs.square();

                let mut rhs = x;
                rhs.square();
                rhs.mul_assign(&x);
                rhs.add_assign(&b_coeff);

                assert!(lhs == rhs);
            }

            let p = <Bn256 as Engine>::G1Affine::from_xy_unchecked(x, y);

            g1_bases.push(p);
        }

        if i == 0 {
            // read G2
            {
                for k in 0..4 {
                    fq_repr.as_mut()[k] = reader.read_u64::<BigEndian>().expect("must read u64");
                }

                let x_c0 = Fq::from_repr(fq_repr).expect("must be valid field element encoding");

                for k in 0..4 {
                    fq_repr.as_mut()[k] = reader.read_u64::<BigEndian>().expect("must read u64");
                }

                let x_c1 = Fq::from_repr(fq_repr).expect("must be valid field element encoding");

                for k in 0..4 {
                    fq_repr.as_mut()[k] = reader.read_u64::<BigEndian>().expect("must read u64");
                }

                let y_c0 = Fq::from_repr(fq_repr).expect("must be valid field element encoding");

                for k in 0..4 {
                    fq_repr.as_mut()[k] = reader.read_u64::<BigEndian>().expect("must read u64");
                }

                let y_c1 = Fq::from_repr(fq_repr).expect("must be valid field element encoding");

                let x = Fq2 { c0: x_c0, c1: x_c1 };

                let y = Fq2 { c0: y_c0, c1: y_c1 };

                {
                    let mut lhs = y;
                    lhs.square();

                    let mut rhs = x;
                    rhs.square();
                    rhs.mul_assign(&x);
                    rhs.add_assign(&b_coeff_fq2);

                    assert!(lhs == rhs);
                }

                let g2 = <Bn256 as Engine>::G2Affine::from_xy_unchecked(x, y);

                g2_bases.push(g2);

                // sanity check by using pairing
                {
                    // check e(g1, g2^x) == e(g1^{x}, g2)
                    let valid = Bn256::final_exponentiation(&Bn256::miller_loop(&[(
                        &g1_bases[0].prepare(),
                        &g2.prepare(),
                    )]))
                    .unwrap()
                        == Bn256::final_exponentiation(&Bn256::miller_loop(&[(
                            &g1_bases[1].prepare(),
                            &g2_bases[0].prepare(),
                        )]))
                        .unwrap();

                    assert!(valid);
                }
            }
            // read G2
            let mut tmp = [0u8; 128];
            reader
                .read_exact(&mut tmp)
                .expect("must skip 128 bytes of irrelevant G2 point");
        }

        // read to end
        reader.consume(64);

        assert_eq!(reader.fill_buf().unwrap().len(), 0);
    }

    assert_eq!(g1_bases.len(), chunk_size * num_chunks + 1);
    assert_eq!(g2_bases.len(), 2);

    let new = Crs::new(g1_bases, g2_bases);

    Ok(new)
}

pub fn make_fflonk_crs_from_ignition_transcripts(
    num_points: usize,
) -> Crs<Bn256, CrsForMonomialForm> {
    let transcripts_dir =
        std::env::var("IGNITION_TRANSCRIPT_PATH").expect("IGNITION_TRANSCRIPT_PATH env variable");
    let chunk_size = 5_040_000usize;
    let num_chunks = num_points.div_ceil(chunk_size);

    // Check transcript files already downloaded from "https://aztec-ignition.s3.eu-west-2.amazonaws.com/MAIN+IGNITION/sealed/transcript{idx}.dat";
    for idx in 0..num_chunks {
        let transcript_file_path = format!("{}/transcript{:02}.dat", transcripts_dir, idx);
        let transcript_file_path = std::path::Path::new(&transcript_file_path);
        assert!(transcript_file_path.exists());
    }

    // transform
    let crs = create_crs_from_ignition_transcript(&transcripts_dir, num_chunks).unwrap();
    let out_path = format!("{}/full_ignition.key", &transcripts_dir);
    let out_file = std::fs::File::create(&out_path).unwrap();

    let bellman::kate_commitment::Crs {
        g1_bases,
        g2_monomial_bases,
        ..
    } = crs;
    assert!(g1_bases.len() >= num_points);
    let mut g1_bases = std::sync::Arc::try_unwrap(g1_bases).unwrap();
    let g2_monomial_bases = std::sync::Arc::try_unwrap(g2_monomial_bases).unwrap();
    g1_bases.truncate(num_points);

    Crs::new(g1_bases, g2_monomial_bases)
}
