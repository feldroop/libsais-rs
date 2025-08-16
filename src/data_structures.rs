use crate::type_model::{OutputElement, SmallAlphabet};

pub trait ResultStructures: sealed::Sealed {}

#[derive(Debug)]
pub struct SuffixArray<O: OutputElement> {
    pub(crate) data: Vec<O>,
}

impl<O: OutputElement> SuffixArray<O> {
    pub fn as_slice(&self) -> &[O] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<O> {
        self.data
    }
}

impl<O: OutputElement> sealed::Sealed for SuffixArray<O> {}

impl<O: OutputElement> ResultStructures for SuffixArray<O> {}

#[derive(Debug)]
pub struct Bwt<I: SmallAlphabet> {
    pub(crate) bwt_data: Vec<I>,
    pub(crate) bwt_primary_index: Option<usize>,
}

impl<I: SmallAlphabet> Bwt<I> {
    pub fn bwt(&self) -> &[I] {
        &self.bwt_data
    }

    pub fn bwt_primary_index(&self) -> Option<usize> {
        self.bwt_primary_index
    }

    pub fn into_parts(self) -> (Vec<I>, Option<usize>) {
        (self.bwt_data, self.bwt_primary_index)
    }
}

impl<I: SmallAlphabet> sealed::Sealed for Bwt<I> {}

impl<I: SmallAlphabet> ResultStructures for Bwt<I> {}

#[derive(Debug)]
pub struct BwtWithAuxIndices<I: SmallAlphabet, O: OutputElement> {
    pub(crate) bwt_data: Vec<I>,
    pub(crate) bwt_primary_index: Option<usize>,
    pub(crate) aux_indices_data: Vec<O>,
}

impl<I: SmallAlphabet, O: OutputElement> BwtWithAuxIndices<I, O> {
    pub fn bwt(&self) -> &[I] {
        &self.bwt_data
    }

    pub fn bwt_primary_index(&self) -> Option<usize> {
        self.bwt_primary_index
    }

    pub fn aux_indices(&self) -> &[O] {
        &self.aux_indices_data
    }

    pub fn into_parts(self) -> (Vec<I>, Option<usize>, Vec<O>) {
        (self.bwt_data, self.bwt_primary_index, self.aux_indices_data)
    }
}

impl<I: SmallAlphabet, O: OutputElement> sealed::Sealed for BwtWithAuxIndices<I, O> {}

impl<I: SmallAlphabet, O: OutputElement> ResultStructures for BwtWithAuxIndices<I, O> {}

mod sealed {
    pub trait Sealed {}
}
