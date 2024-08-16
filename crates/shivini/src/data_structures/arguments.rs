use super::*;
use crate::data_structures::cache::StorageCache;
use boojum::cs::LookupParameters;
use std::ops::Range;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArgumentsPolyType {
    Z,
    PartialProduct,
    LookupA,
    LookupB,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArgumentsLayout {
    pub num_z_polys: usize,
    pub num_partial_products: usize,
    pub num_lookup_a_polys: usize,
    pub num_lookup_b_polys: usize,
}

impl ArgumentsLayout {
    pub fn from_trace_layout_and_lookup_params(
        trace_layout: TraceLayout,
        quotient_degree: usize,
        lookup_params: LookupParameters,
    ) -> Self {
        let num_z_polys = 1;
        let num_variable_cols = trace_layout.num_variable_cols;
        let mut num_partial_products = num_variable_cols / quotient_degree;
        if num_variable_cols % quotient_degree != 0 {
            num_partial_products += 1;
        }
        num_partial_products -= 1; // ignore last partial product

        let (num_lookup_a_polys, num_lookup_b_polys) =
            if lookup_params == LookupParameters::NoLookup {
                (0, 0)
            } else {
                match lookup_params {
                    LookupParameters::UseSpecializedColumnsWithTableIdAsVariable {
                        width: _,
                        num_repetitions,
                        share_table_id,
                    } => {
                        assert!(!share_table_id);
                        (num_repetitions, 1)
                    }
                    LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
                        width: _,
                        num_repetitions,
                        share_table_id,
                    } => {
                        assert!(share_table_id);
                        (num_repetitions, 1)
                    }
                    _ => unreachable!(),
                }
            };

        let num_z_polys = num_z_polys * 2;
        let num_partial_products = num_partial_products * 2;
        let num_lookup_a_polys = num_lookup_a_polys * 2;
        let num_lookup_b_polys = num_lookup_b_polys * 2;

        Self {
            num_z_polys,
            num_partial_products,
            num_lookup_a_polys,
            num_lookup_b_polys,
        }
    }

    pub fn num_polys(&self) -> usize {
        self.num_z_polys
            + self.num_partial_products
            + self.num_lookup_a_polys
            + self.num_lookup_b_polys
    }
}

impl GenericPolynomialStorageLayout for ArgumentsLayout {
    type PolyType = ArgumentsPolyType;

    fn num_polys(&self) -> usize {
        self.num_polys()
    }

    fn poly_range(&self, poly_type: Self::PolyType) -> (Range<usize>, Self) {
        let start = match poly_type {
            ArgumentsPolyType::Z => 0,
            ArgumentsPolyType::PartialProduct => self.num_z_polys,
            ArgumentsPolyType::LookupA => self.num_z_polys + self.num_partial_products,
            ArgumentsPolyType::LookupB => {
                self.num_z_polys + self.num_partial_products + self.num_lookup_a_polys
            }
        };
        let len = match poly_type {
            ArgumentsPolyType::Z => self.num_z_polys,
            ArgumentsPolyType::PartialProduct => self.num_partial_products,
            ArgumentsPolyType::LookupA => self.num_lookup_a_polys,
            ArgumentsPolyType::LookupB => self.num_lookup_b_polys,
        };
        let range = start..start + len;
        let layout = Self {
            num_z_polys: match poly_type {
                ArgumentsPolyType::Z => self.num_z_polys,
                _ => 0,
            },
            num_partial_products: match poly_type {
                ArgumentsPolyType::PartialProduct => self.num_partial_products,
                _ => 0,
            },
            num_lookup_a_polys: match poly_type {
                ArgumentsPolyType::LookupA => self.num_lookup_a_polys,
                _ => 0,
            },
            num_lookup_b_polys: match poly_type {
                ArgumentsPolyType::LookupB => self.num_lookup_b_polys,
                _ => 0,
            },
        };
        (range, layout)
    }
}

pub type GenericArgumentsStorage<P> = GenericStorage<P, ArgumentsLayout>;

pub type ArgumentsCache<H> = StorageCache<ArgumentsLayout, H>;

pub struct ArgumentsPolynomials<'a, P: PolyForm> {
    pub z_polys: Vec<ComplexPoly<'a, P>>,
    pub partial_products: Vec<ComplexPoly<'a, P>>,
    pub lookup_a_polys: Vec<ComplexPoly<'a, P>>,
    pub lookup_b_polys: Vec<ComplexPoly<'a, P>>,
}

impl<'a, P: PolyForm> ArgumentsPolynomials<'a, P> {
    pub fn new(mut polynomials: Vec<ComplexPoly<'a, P>>, layout: ArgumentsLayout) -> Self {
        let ArgumentsLayout {
            num_z_polys,
            num_partial_products,
            num_lookup_a_polys,
            num_lookup_b_polys,
        } = layout;
        assert_eq!(num_z_polys % 2, 0);
        let num_z_polys = num_z_polys / 2;
        assert_eq!(num_partial_products % 2, 0);
        let num_partial_products = num_partial_products / 2;
        assert_eq!(num_lookup_a_polys % 2, 0);
        let num_lookup_a_polys = num_lookup_a_polys / 2;
        assert_eq!(num_lookup_b_polys % 2, 0);
        let num_lookup_b_polys = num_lookup_b_polys / 2;
        let lookup_b_polys = polynomials.split_off(polynomials.len() - num_lookup_b_polys);
        let lookup_a_polys = polynomials.split_off(polynomials.len() - num_lookup_a_polys);
        let partial_products = polynomials.split_off(polynomials.len() - num_partial_products);
        let z_polys = polynomials.split_off(polynomials.len() - num_z_polys);
        assert!(polynomials.is_empty());
        Self {
            z_polys,
            partial_products,
            lookup_a_polys,
            lookup_b_polys,
        }
    }
}

impl<P: PolyForm> GenericArgumentsStorage<P> {
    pub fn as_polynomials(&self) -> ArgumentsPolynomials<P> {
        let layout = self.layout;
        let polynomials = self.as_complex_polys();
        ArgumentsPolynomials::new(polynomials, layout)
    }

    pub fn as_polynomials_mut(&mut self) -> ArgumentsPolynomials<P> {
        let layout = self.layout;
        let polynomials = self.as_complex_polys_mut();
        ArgumentsPolynomials::new(polynomials, layout)
    }
}

impl GenericArgumentsStorage<CosetEvaluations> {
    pub(crate) fn barycentric_evaluate(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF>> {
        batch_barycentric_evaluate_ext(self, bases, self.domain_size(), self.num_polys() / 2)
    }
}
