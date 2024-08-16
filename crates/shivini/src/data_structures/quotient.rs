use super::*;
use crate::data_structures::AsSingleSlice;
use crate::poly::CosetEvaluations;
use era_cudart::result::CudaResult;
use std::ops::Range;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum QuotientPolyType {
    Quotient,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct QuotientLayout {
    pub num_quotient_polys: usize,
}

impl QuotientLayout {
    pub fn new(quotient_degree: usize) -> Self {
        Self {
            num_quotient_polys: quotient_degree << 1,
        }
    }

    pub fn num_polys(&self) -> usize {
        self.num_quotient_polys
    }
}

impl GenericPolynomialStorageLayout for QuotientLayout {
    type PolyType = QuotientPolyType;

    fn num_polys(&self) -> usize {
        self.num_polys()
    }

    fn poly_range(&self, _poly_type: Self::PolyType) -> (Range<usize>, Self) {
        let num_quotient_polys = self.num_quotient_polys;
        (0..num_quotient_polys, QuotientLayout { num_quotient_polys })
    }
}

pub type GenericQuotientStorage<P> = GenericStorage<P, QuotientLayout>;

pub struct QuotientPolynomials<'a, P: PolyForm> {
    pub quotient_polys: Vec<ComplexPoly<'a, P>>,
}

impl<'a, P: PolyForm> QuotientPolynomials<'a, P> {
    pub fn new(polynomials: Vec<ComplexPoly<'a, P>>, layout: QuotientLayout) -> Self {
        let QuotientLayout { num_quotient_polys } = layout;
        assert_eq!(num_quotient_polys % 2, 0);
        let num_quotient_polys = num_quotient_polys / 2;
        assert_eq!(num_quotient_polys, polynomials.len());
        Self {
            quotient_polys: polynomials,
        }
    }
}

impl<P: PolyForm> GenericQuotientStorage<P> {
    pub fn as_polynomials(&self) -> QuotientPolynomials<P> {
        let layout = self.layout;
        let polynomials = self.as_complex_polys();
        QuotientPolynomials::new(polynomials, layout)
    }

    pub fn as_polynomials_mut(&mut self) -> QuotientPolynomials<P> {
        let layout = self.layout;
        let polynomials = self.as_complex_polys_mut();
        QuotientPolynomials::new(polynomials, layout)
    }
}

impl GenericQuotientStorage<CosetEvaluations> {
    pub(crate) fn barycentric_evaluate(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF>> {
        batch_barycentric_evaluate_ext(self, bases, self.domain_size(), self.num_polys() / 2)
    }
}

pub type QuotientCache<H> = StorageCache<QuotientLayout, H>;
