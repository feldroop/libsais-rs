use either::Either;

use crate::{
    ThreadCount,
    construction::{AuxIndicesSamplingRate, lcp::LcpConstruction, plcp::PlcpConstruction},
    type_model::{
        BufferMode, InputElement, IsValidOutputFor, OutputElement, OwnedBuffer, SingleThreaded,
        SmallAlphabet, SupportsPlcpOutputFor,
    },
};

use std::marker::PhantomData;

// -------------------- suffix array with text --------------------
#[derive(Debug)]
pub struct SuffixArrayWithText<'s, 't, I: InputElement, O: OutputElement, B: BufferMode> {
    pub(crate) suffix_array: OwnedOrBorrowed<'s, O, B>,
    pub(crate) text: &'t [I],
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'s, 't, I: InputElement, O: OutputElement, B: BufferMode>
    SuffixArrayWithText<'s, 't, I, O, B>
{
    pub fn suffix_array(&self) -> &[O] {
        &self.suffix_array.buffer
    }

    pub fn text(&self) -> &'t [I] {
        self.text
    }

    pub fn is_generalized_suffix_array(&self) -> bool {
        self.is_generalized_suffix_array
    }
}

impl<'t, I: InputElement, O: OutputElement> SuffixArrayWithText<'static, 't, I, O, OwnedBuffer> {
    pub fn into_vec(self) -> Vec<O> {
        self.suffix_array.into_inner()
    }
}

impl<'s, 't, I: InputElement, O: OutputElement, B: BufferMode>
    SuffixArrayWithText<'s, 't, I, O, B>
{
    pub fn into_parts(self) -> (B::Buffer<'s, O>, &'t [I], bool) {
        (
            self.suffix_array.into_inner(),
            self.text,
            self.is_generalized_suffix_array,
        )
    }
}

impl<'s, 't, I: InputElement, O: IsValidOutputFor<I>, B: BufferMode>
    SuffixArrayWithText<'s, 't, I, O, B>
{
    pub unsafe fn from_parts(
        suffix_array: B::Buffer<'s, O>,
        text: &'t [I],
        is_generalized_suffix_array: bool,
    ) -> Self {
        Self {
            suffix_array: OwnedOrBorrowed::new(suffix_array),
            text,
            is_generalized_suffix_array,
        }
    }
}

impl<'s, 't, I: InputElement, O: SupportsPlcpOutputFor<I>, SaB: BufferMode>
    SuffixArrayWithText<'s, 't, I, O, SaB>
{
    pub fn plcp_construction(
        self,
    ) -> PlcpConstruction<'static, 's, 't, I, O, OwnedBuffer, SaB, SingleThreaded> {
        PlcpConstruction {
            text: self.text,
            suffix_array_buffer: self.suffix_array,
            generalized_suffix_array: self.is_generalized_suffix_array,
            plcp_buffer: None,
            thread_count: ThreadCount::fixed(1),
            _parallelism_marker: PhantomData,
            _plcp_buffer_mode_marker: PhantomData,
        }
    }
}

// -------------------- bwt (with aux indices) --------------------
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

// -------------------- suffix array with plcp --------------------
pub struct SuffixArrayWithPlcp<'p, 's, O: OutputElement, PlcpB: BufferMode, SaB: BufferMode> {
    pub(crate) plcp: OwnedOrBorrowed<'p, O, PlcpB>,
    pub(crate) suffix_array: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'p, 's, O: OutputElement, PlcpB: BufferMode, SaB: BufferMode>
    SuffixArrayWithPlcp<'p, 's, O, PlcpB, SaB>
{
    pub fn plcp(&self) -> &[O] {
        &self.plcp.buffer
    }

    pub fn suffix_array(&self) -> &[O] {
        &self.suffix_array.buffer
    }

    pub fn is_generalized_suffix_array(&self) -> bool {
        self.is_generalized_suffix_array
    }

    pub fn into_parts(self) -> (SaB::Buffer<'s, O>, PlcpB::Buffer<'p, O>, bool) {
        (
            self.suffix_array.into_inner(),
            self.plcp.into_inner(),
            self.is_generalized_suffix_array,
        )
    }

    pub unsafe fn from_parts(
        plcp: PlcpB::Buffer<'p, O>,
        suffix_array: SaB::Buffer<'s, O>,
        is_generalized_suffix_array: bool,
    ) -> Self {
        Self {
            plcp: OwnedOrBorrowed::new(plcp),
            suffix_array: OwnedOrBorrowed::new(suffix_array),
            is_generalized_suffix_array,
        }
    }

    pub fn lcp_construction(
        self,
    ) -> LcpConstruction<'static, 'p, 's, O, OwnedBuffer, PlcpB, SaB, SingleThreaded> {
        LcpConstruction {
            plcp_buffer: self.plcp,
            suffix_array_buffer: self.suffix_array,
            lcp_buffer: None,
            thread_count: ThreadCount::fixed(1),
            is_generalized_suffix_array: self.is_generalized_suffix_array,
            _parallelism_marker: PhantomData,
            _lcp_buffer_mode_marker: PhantomData,
        }
    }
}

pub struct SuffixArrayWithLcpAndPlcp<
    'l,
    'p,
    's,
    O: OutputElement,
    LcpB: BufferMode,
    PlcpB: BufferMode,
    SaB: BufferMode,
> {
    pub(crate) lcp: OwnedOrBorrowed<'l, O, LcpB>,
    pub(crate) plcp: OwnedOrBorrowed<'p, O, PlcpB>,
    pub(crate) suffix_array: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'l, 'p, 's, O: OutputElement, LcpB: BufferMode, PlcpB: BufferMode, SaB: BufferMode>
    SuffixArrayWithLcpAndPlcp<'l, 'p, 's, O, LcpB, PlcpB, SaB>
{
    pub fn lcp(&self) -> &[O] {
        &self.lcp.buffer
    }

    pub fn plcp(&self) -> &[O] {
        &self.plcp.buffer
    }

    pub fn suffix_array(&self) -> &[O] {
        &self.suffix_array.buffer
    }

    pub fn is_generalized_suffix_array(&self) -> bool {
        self.is_generalized_suffix_array
    }

    pub fn into_parts(
        self,
    ) -> (
        SaB::Buffer<'s, O>,
        LcpB::Buffer<'l, O>,
        PlcpB::Buffer<'p, O>,
        bool,
    ) {
        (
            self.suffix_array.into_inner(),
            self.lcp.into_inner(),
            self.plcp.into_inner(),
            self.is_generalized_suffix_array,
        )
    }
}

pub struct LcpAndPlcp<'l, 'p, O: OutputElement, LcpB: BufferMode, PlcpB: BufferMode> {
    pub(crate) lcp: OwnedOrBorrowed<'l, O, LcpB>,
    pub(crate) plcp: OwnedOrBorrowed<'p, O, PlcpB>,
}

impl<'l, 'p, O: OutputElement, LcpB: BufferMode, PlcpB: BufferMode>
    LcpAndPlcp<'l, 'p, O, LcpB, PlcpB>
{
    pub fn lcp(&self) -> &[O] {
        &self.lcp.buffer
    }

    pub fn plcp(&self) -> &[O] {
        &self.plcp.buffer
    }

    pub fn into_parts(self) -> (LcpB::Buffer<'l, O>, PlcpB::Buffer<'p, O>) {
        (self.lcp.into_inner(), self.plcp.into_inner())
    }
}

// -------------------- util struct --------------------
#[derive(Debug)]
pub(crate) struct OwnedOrBorrowed<'a, T, B> {
    pub(crate) buffer: Either<Vec<T>, &'a mut [T]>,
    pub(crate) _marker: PhantomData<B>,
}

impl<'a, T: std::fmt::Debug, B: BufferMode> OwnedOrBorrowed<'a, T, B> {
    pub(crate) fn new(buffer: B::Buffer<'a, T>) -> OwnedOrBorrowed<'a, T, B> {
        OwnedOrBorrowed {
            buffer: B::buffer_to_either(buffer),
            _marker: PhantomData,
        }
    }

    pub(crate) fn into_inner(self) -> B::Buffer<'a, T> {
        B::either_to_buffer(self.buffer)
    }
}

impl<'a, T, B: BufferMode> OwnedOrBorrowed<'a, T, B> {
    pub(crate) fn shorten_buffer_to(&mut self, len: usize) {
        match &mut self.buffer {
            Either::Left(owned) => crate::construction::free_extra_space(owned, len),
            Either::Right(borrowed) => *borrowed = &mut std::mem::take(borrowed)[..len],
        }
    }
}

impl<'a, T, B: BufferMode> OwnedOrBorrowed<'a, T, B>
where
    T: TryFrom<usize, Error: std::fmt::Debug> + Clone + std::fmt::Debug,
{
    pub(crate) fn take_buffer_or_allocate<F: FnOnce() -> Vec<T>>(
        opt: Option<&'a mut [T]>,
        f: F,
    ) -> OwnedOrBorrowed<'a, T, B> {
        Self::new(B::unwrap_or_allocate(opt, f))
    }
}

impl<'a, T, B1: BufferMode> OwnedOrBorrowed<'a, T, B1> {
    pub(crate) fn into_other_marker_type<B2: BufferMode>(self) -> OwnedOrBorrowed<'a, T, B2> {
        OwnedOrBorrowed {
            buffer: self.buffer,
            _marker: PhantomData,
        }
    }
}
