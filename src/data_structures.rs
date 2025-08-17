use crate::{
    construction::AuxIndicesSamplingRate,
    type_model::{OutputElementDecided, SmallAlphabet},
};

#[derive(Debug)]
pub struct SuffixArray<O: OutputElementDecided> {
    pub(crate) data: Vec<O>,
}

impl<O: OutputElementDecided> SuffixArray<O> {
    pub fn as_slice(&self) -> &[O] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<O> {
        self.data
    }
}

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

#[derive(Debug)]
pub struct AuxIndices<O: OutputElementDecided> {
    pub(crate) data: Vec<O>,
    pub(crate) sampling_rate: AuxIndicesSamplingRate<O>,
}

impl<O: OutputElementDecided> AuxIndices<O> {
    pub fn as_slice(&self) -> &[O] {
        &self.data
    }

    pub fn sampling_rate(&self) -> AuxIndicesSamplingRate<O> {
        self.sampling_rate
    }

    pub fn into_parts(self) -> (Vec<O>, AuxIndicesSamplingRate<O>) {
        (self.data, self.sampling_rate)
    }
}

#[derive(Debug)]
pub struct BwtWithAuxIndices<I: SmallAlphabet, O: OutputElementDecided> {
    pub(crate) bwt_data: Vec<I>,
    pub(crate) aux_indices: AuxIndices<O>,
}

impl<I: SmallAlphabet, O: OutputElementDecided> BwtWithAuxIndices<I, O> {
    pub fn bwt(&self) -> &[I] {
        &self.bwt_data
    }

    pub fn aux_indices(&self) -> &[O] {
        self.aux_indices.as_slice()
    }

    pub fn into_parts(self) -> (Vec<I>, Vec<O>, AuxIndicesSamplingRate<O>) {
        let (aux_indices_data, aux_indices_sampling_rate) = self.aux_indices.into_parts();
        (self.bwt_data, aux_indices_data, aux_indices_sampling_rate)
    }
}
