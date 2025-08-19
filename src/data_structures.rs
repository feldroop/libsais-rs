use crate::{
    ThreadCount,
    construction::{AuxIndicesSamplingRate, plcp::PlcpConstruction},
    type_model::{
        InputElement, IsValidOutputFor, OutputElement, OwnedBuffer, SingleThreaded, SmallAlphabet,
        SupportsPlcpOutputFor,
    },
};

use std::marker::PhantomData;

#[derive(Debug)]
pub struct SuffixArrayWithText<'t, I: InputElement, O: OutputElement> {
    pub(crate) suffix_array: Vec<O>,
    pub(crate) text: &'t [I],
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'t, I: InputElement, O: OutputElement> SuffixArrayWithText<'t, I, O> {
    pub fn suffix_array(&self) -> &[O] {
        &self.suffix_array
    }

    pub fn text(&self) -> &'t [I] {
        self.text
    }

    pub fn into_vec(self) -> Vec<O> {
        self.suffix_array
    }

    pub fn is_generalized_suffix_array(&self) -> bool {
        self.is_generalized_suffix_array
    }

    pub fn into_parts(self) -> (Vec<O>, &'t [I], bool) {
        (
            self.suffix_array,
            self.text,
            self.is_generalized_suffix_array,
        )
    }
}

impl<'t, I: InputElement, O: SupportsPlcpOutputFor<I>> SuffixArrayWithText<'t, I, O> {
    pub fn plcp_construction<'s>(
        &'s self,
    ) -> PlcpConstruction<'static, 's, 't, I, O, OwnedBuffer, SingleThreaded> {
        PlcpConstruction {
            text: self.text,
            suffix_array_buffer: &self.suffix_array,
            generalized_suffix_array: self.is_generalized_suffix_array,
            plcp_buffer: None,
            thread_count: ThreadCount::fixed(1),
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
        }
    }
}

impl<'t, I: InputElement, O: IsValidOutputFor<I>> SuffixArrayWithText<'t, I, O> {
    pub unsafe fn from_parts(
        suffix_array: Vec<O>,
        text: &'t [I],
        is_generalized_suffix_array: bool,
    ) -> Self {
        Self {
            suffix_array,
            text,
            is_generalized_suffix_array,
        }
    }
}

#[derive(Debug)]
pub struct BorrowedSuffixArrayWithText<'s, 't, I: InputElement, O: OutputElement> {
    pub(crate) suffix_array_buffer: &'s [O],
    pub(crate) text: &'t [I],
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'s, 't, I: InputElement, O: OutputElement> BorrowedSuffixArrayWithText<'s, 't, I, O> {
    pub fn suffix_array(&self) -> &'s [O] {
        &self.suffix_array_buffer
    }

    pub fn text(&self) -> &'t [I] {
        self.text
    }

    pub fn is_generalized_suffix_array(&self) -> bool {
        self.is_generalized_suffix_array
    }

    pub fn into_parts(self) -> (&'s [O], &'t [I], bool) {
        (
            self.suffix_array_buffer,
            self.text,
            self.is_generalized_suffix_array,
        )
    }
}

impl<'s, 't, I: InputElement, O: SupportsPlcpOutputFor<I>>
    BorrowedSuffixArrayWithText<'s, 't, I, O>
{
    pub fn plcp_construction(
        &self,
    ) -> PlcpConstruction<'static, 's, 't, I, O, OwnedBuffer, SingleThreaded> {
        PlcpConstruction {
            text: &self.text,
            suffix_array_buffer: &self.suffix_array_buffer,
            generalized_suffix_array: self.is_generalized_suffix_array,
            plcp_buffer: None,
            thread_count: ThreadCount::fixed(1),
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
        }
    }
}

impl<'s, 't, I: InputElement, O: IsValidOutputFor<I>> BorrowedSuffixArrayWithText<'s, 't, I, O> {
    pub unsafe fn from_parts(
        suffix_array_buffer: &'s [O],
        text: &'t [I],
        is_generalized_suffix_array: bool,
    ) -> Self {
        Self {
            suffix_array_buffer,
            text,
            is_generalized_suffix_array,
        }
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
pub struct AuxIndices<O: OutputElement> {
    pub(crate) data: Vec<O>,
    pub(crate) sampling_rate: AuxIndicesSamplingRate<O>,
}

impl<O: OutputElement> AuxIndices<O> {
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
pub struct BwtWithAuxIndices<I: SmallAlphabet, O: OutputElement> {
    pub(crate) bwt_data: Vec<I>,
    pub(crate) aux_indices: AuxIndices<O>,
}

impl<I: SmallAlphabet, O: OutputElement> BwtWithAuxIndices<I, O> {
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

pub struct PlcpWithBorrowedSuffixArray<'s, O: OutputElement> {
    pub(crate) plcp: Vec<O>,
    pub(crate) suffix_array: &'s [O],
}

impl<'s, O: OutputElement> PlcpWithBorrowedSuffixArray<'s, O> {
    pub fn plcp(&self) -> &[O] {
        &self.plcp
    }

    pub fn suffix_array(&self) -> &'s [O] {
        self.suffix_array
    }

    pub fn into_parts(self) -> (Vec<O>, &'s [O]) {
        (self.plcp, self.suffix_array)
    }

    pub unsafe fn from_parts(plcp: Vec<O>, suffix_array: &'s [O]) -> Self {
        Self { plcp, suffix_array }
    }

    // TODO construct lcp
}

pub struct BorrowedPlcpWithBorrowedSuffixArray<'p, 's, O: OutputElement> {
    pub(crate) plcp: &'p [O],
    pub(crate) suffix_array: &'s [O],
}

impl<'p, 's, O: OutputElement> BorrowedPlcpWithBorrowedSuffixArray<'p, 's, O> {
    pub fn plcp(&self) -> &'p [O] {
        self.plcp
    }

    pub fn suffix_array(&self) -> &'s [O] {
        self.suffix_array
    }

    pub fn into_parts(self) -> (&'p [O], &'s [O]) {
        (self.plcp, self.suffix_array)
    }

    pub unsafe fn from_parts(plcp: &'p [O], suffix_array: &'s [O]) -> Self {
        Self { plcp, suffix_array }
    }

    // TODO construct lcp
}
