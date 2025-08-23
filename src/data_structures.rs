use either::Either;

use crate::{
    ThreadCount,
    construction::{
        AuxIndicesSamplingRate, lcp::LcpConstruction, plcp::PlcpConstruction, unbwt::UnBwt,
    },
    type_model::{
        BufferMode, InputElement, IsValidOutputFor, OutputElement, OwnedBuffer, SingleThreaded,
        SmallAlphabet, SupportsPlcpOutputFor, Undecided,
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
pub struct Bwt<'b, I: SmallAlphabet, B: BufferMode> {
    pub(crate) bwt: OwnedOrBorrowed<'b, I, B>,
    pub(crate) primary_index: usize,
}

impl<'b, I: SmallAlphabet, B: BufferMode> Bwt<'b, I, B> {
    pub fn bwt(&self) -> &[I] {
        &self.bwt.buffer
    }

    pub fn primary_index(&self) -> usize {
        self.primary_index
    }

    pub fn into_parts(self) -> (B::Buffer<'b, I>, usize) {
        (self.bwt.into_inner(), self.primary_index)
    }

    pub unsafe fn from_parts(bwt: B::Buffer<'b, I>, primary_index: usize) -> Self {
        Self {
            bwt: OwnedOrBorrowed::new(bwt),
            primary_index,
        }
    }

    pub fn unbwt(self) -> UnBwt<'b, 'static, 'static, I, Undecided, B, OwnedBuffer, Undecided> {
        UnBwt {
            bwt: Some(self.bwt),
            text: None,
            temporary_array_buffer: None,
            frequency_table: None,
            thread_count: ThreadCount::fixed(1),
            context: None,
            primary_index: Some(self.primary_index),
            aux_indices_sampling_rate: None,
            aux_indices_buffer: None,
            _text_buffer_mode_marker: PhantomData,
        }
    }
}

impl<I: SmallAlphabet> Bwt<'static, I, OwnedBuffer> {
    pub fn into_vec(self) -> Vec<I> {
        self.bwt.into_inner()
    }
}

#[derive(Debug)]
pub struct BwtWithAuxIndices<
    'a,
    'b,
    I: SmallAlphabet,
    O: OutputElement,
    AuxB: BufferMode,
    BwtB: BufferMode,
> {
    pub(crate) bwt: OwnedOrBorrowed<'b, I, BwtB>,
    pub(crate) aux_indices: OwnedOrBorrowed<'a, O, AuxB>,
    pub(crate) aux_indices_sampling_rate: AuxIndicesSamplingRate,
}

impl<'a, 'b, I: SmallAlphabet, O: OutputElement, AuxB: BufferMode, BwtB: BufferMode>
    BwtWithAuxIndices<'a, 'b, I, O, AuxB, BwtB>
{
    pub fn bwt(&self) -> &[I] {
        &self.bwt.buffer
    }

    pub fn aux_indices(&self) -> &[O] {
        &self.aux_indices.buffer
    }

    pub fn aux_indices_sampling_rate(&self) -> AuxIndicesSamplingRate {
        self.aux_indices_sampling_rate
    }

    pub fn into_parts(
        self,
    ) -> (
        BwtB::Buffer<'b, I>,
        AuxB::Buffer<'a, O>,
        AuxIndicesSamplingRate,
    ) {
        (
            self.bwt.into_inner(),
            self.aux_indices.into_inner(),
            self.aux_indices_sampling_rate,
        )
    }

    pub unsafe fn from_parts(
        bwt: BwtB::Buffer<'b, I>,
        aux_indices: AuxB::Buffer<'a, O>,
        aux_indices_sampling_rate: AuxIndicesSamplingRate,
    ) -> Self {
        Self {
            bwt: OwnedOrBorrowed::new(bwt),
            aux_indices: OwnedOrBorrowed::new(aux_indices),
            aux_indices_sampling_rate,
        }
    }

    pub fn unbwt(self) -> UnBwt<'b, 'a, 'static, I, O, BwtB, OwnedBuffer, Undecided> {
        UnBwt {
            bwt: Some(self.bwt),
            text: None,
            temporary_array_buffer: None,
            frequency_table: None,
            thread_count: ThreadCount::fixed(1),
            context: None,
            primary_index: None,
            aux_indices_sampling_rate: Some(self.aux_indices_sampling_rate),
            aux_indices_buffer: Some(self.aux_indices.buffer),
            _text_buffer_mode_marker: PhantomData,
        }
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
pub struct Text<'t, I: InputElement, B: BufferMode> {
    pub(crate) text: OwnedOrBorrowed<'t, I, B>,
}

impl<'t, I: InputElement, B: BufferMode> Text<'t, I, B> {
    pub fn as_slice(&self) -> &[I] {
        &self.text.buffer
    }

    pub fn into_inner(self) -> B::Buffer<'t, I> {
        self.text.into_inner()
    }
}

impl<'t, I: InputElement> Text<'t, I, OwnedBuffer> {
    pub fn into_vec(self) -> Vec<I> {
        self.text.into_inner()
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
