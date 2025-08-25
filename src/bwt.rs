use either::Either;

use std::marker::PhantomData;

use crate::{
    OutputElement, SmallAlphabet, ThreadCount,
    context::Context,
    error::{IntoSaisResult, LibsaisError},
    generics_dispatch::{LibsaisFunctionsSmallAlphabet, SmallAlphabetFunctionsDispatch},
    owned_or_borrowed::OwnedOrBorrowed,
    suffix_array::{self, ExtraSpace},
    type_state::{
        AuxIndicesMode, BorrowedBuffer, BufferMode, BufferModeOrUndecided, NoAuxIndices,
        OutputElementOrUndecided, OwnedBuffer, Parallelism, ParallelismOrUndecided, SingleThreaded,
        Undecided,
    },
    unbwt::UnBwt,
};

#[cfg(feature = "openmp")]
use crate::type_state::MultiThreaded;

#[derive(Debug)]
pub struct BwtConstruction<
    'a,
    'b,
    'r,
    I: SmallAlphabet,
    O: OutputElementOrUndecided,
    B: BufferModeOrUndecided,
    P: ParallelismOrUndecided,
    A: AuxIndicesMode,
> {
    text: Option<&'r [I]>,
    bwt_buffer: Option<&'b mut [I]>,
    temporary_array_buffer: Option<&'r mut [O]>,
    frequency_table: Option<&'r mut [O]>,
    extra_space_temporary_array_buffer: ExtraSpace,
    thread_count: ThreadCount,
    context: Option<&'r mut Context<I, O, P>>,
    aux_indices_sampling_rate: Option<AuxIndicesSamplingRate<O>>,
    aux_indices_buffer: Option<&'a mut [O]>,
    _buffer_mode_marker: PhantomData<B>,
    _aux_indices_mode_marker: PhantomData<A>,
}

impl<
    'a,
    'b,
    'r,
    I: SmallAlphabet,
    O: OutputElementOrUndecided,
    B: BufferModeOrUndecided,
    P: ParallelismOrUndecided,
    A: AuxIndicesMode,
> BwtConstruction<'a, 'b, 'r, I, O, B, P, A>
{
    fn init() -> Self {
        Self {
            text: None,
            bwt_buffer: None,
            temporary_array_buffer: None,
            frequency_table: None,
            thread_count: ThreadCount::fixed(1),
            extra_space_temporary_array_buffer: ExtraSpace::Recommended,
            context: None,
            aux_indices_sampling_rate: None,
            aux_indices_buffer: None,
            _buffer_mode_marker: PhantomData,
            _aux_indices_mode_marker: PhantomData,
        }
    }
}

impl<
    'a,
    'b,
    'r,
    I: SmallAlphabet,
    O: OutputElementOrUndecided,
    B1: BufferModeOrUndecided,
    P: ParallelismOrUndecided,
    A1: AuxIndicesMode,
> BwtConstruction<'a, 'b, 'r, I, O, B1, P, A1>
{
    fn into_other_marker_type<B2: BufferModeOrUndecided, A2: AuxIndicesMode>(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B2, P, A2> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_array_buffer: self.temporary_array_buffer,
            frequency_table: self.frequency_table,
            thread_count: self.thread_count,
            extra_space_temporary_array_buffer: self.extra_space_temporary_array_buffer,
            context: self.context,
            aux_indices_sampling_rate: self.aux_indices_sampling_rate,
            aux_indices_buffer: self.aux_indices_buffer,
            _buffer_mode_marker: PhantomData,
            _aux_indices_mode_marker: PhantomData,
        }
    }
}

// entry point to builder
impl<'a, 'b, 'r, I: SmallAlphabet>
    BwtConstruction<'a, 'b, 'r, I, Undecided, Undecided, Undecided, NoAuxIndices>
{
    pub fn for_text(text: &'r [I]) -> Self {
        Self {
            text: Some(text),
            ..Self::init()
        }
    }

    pub fn replace_text(
        text: &'b mut [I],
    ) -> BwtConstruction<'a, 'b, 'r, I, Undecided, BorrowedBuffer, Undecided, NoAuxIndices> {
        BwtConstruction {
            bwt_buffer: Some(text),
            ..BwtConstruction::init()
        }
    }

    pub fn in_borrowed_buffer(
        self,
        bwt_buffer: &'b mut [I],
    ) -> BwtConstruction<'a, 'b, 'r, I, Undecided, BorrowedBuffer, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: Some(bwt_buffer),
            ..BwtConstruction::init()
        }
    }

    pub fn in_owned_buffer(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, Undecided, OwnedBuffer, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            ..BwtConstruction::init()
        }
    }
}

// second choice: output type
impl<'a, 'b, 'r, I: SmallAlphabet, B: BufferMode>
    BwtConstruction<'a, 'b, 'r, I, Undecided, B, Undecided, NoAuxIndices>
{
    pub fn with_borrowed_temporary_array_buffer<O: OutputElement>(
        self,
        temporary_array_buffer: &'r mut [O],
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_array_buffer: Some(temporary_array_buffer),
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_array_buffer<O: OutputElement>(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_array_buffer32(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, i32, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_array_buffer64(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, i64, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_array_buffer_and_extra_space<O: OutputElement>(
        self,
        extra_space: ExtraSpace,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            extra_space_temporary_array_buffer: extra_space,
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_array_buffer_and_extra_space32(
        self,
        extra_space: ExtraSpace,
    ) -> BwtConstruction<'a, 'b, 'r, I, i32, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            extra_space_temporary_array_buffer: extra_space,
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_array_buffer_and_extra_space64(
        self,
        extra_space: ExtraSpace,
    ) -> BwtConstruction<'a, 'b, 'r, I, i64, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            extra_space_temporary_array_buffer: extra_space,
            ..BwtConstruction::init()
        }
    }
}

// third choice: threading
impl<'a, 'b, 'r, I: SmallAlphabet, O: OutputElement, B: BufferMode>
    BwtConstruction<'a, 'b, 'r, I, O, B, Undecided, NoAuxIndices>
{
    pub fn single_threaded(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, SingleThreaded, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_array_buffer: self.temporary_array_buffer,
            extra_space_temporary_array_buffer: self.extra_space_temporary_array_buffer,
            ..BwtConstruction::init()
        }
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        self,
        thread_count: ThreadCount,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, MultiThreaded, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_array_buffer: self.temporary_array_buffer,
            extra_space_temporary_array_buffer: self.extra_space_temporary_array_buffer,
            thread_count,
            ..BwtConstruction::init()
        }
    }
}

// optional choice at any time: with auxiliary indices
impl<'a, 'b, 'r, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    BwtConstruction<'a, 'b, 'r, I, O, B, P, NoAuxIndices>
{
    pub fn with_aux_indices(
        mut self,
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, P, OwnedBuffer> {
        self.aux_indices_sampling_rate = Some(aux_indices_sampling_rate);
        self.into_other_marker_type()
    }

    pub fn with_aux_indices_in_buffer(
        mut self,
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
        aux_indices_buffer: &'a mut [O],
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, P, BorrowedBuffer> {
        self.aux_indices_sampling_rate = Some(aux_indices_sampling_rate);
        self.aux_indices_buffer = Some(aux_indices_buffer);
        self.into_other_marker_type()
    }
}

impl<'a, 'b, 'r, I: SmallAlphabet, B: BufferMode, P: Parallelism, A: AuxIndicesMode>
    BwtConstruction<'a, 'b, 'r, I, i32, B, P, A>
{
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded 32-bit output version.
    pub fn with_context(self, context: &'r mut Context<I, i32, P>) -> Self {
        Self {
            context: Some(context),
            ..self
        }
    }
}

impl<
    'a,
    'b,
    'r,
    I: SmallAlphabet,
    O: OutputElement,
    B: BufferMode,
    P: Parallelism,
    A: AuxIndicesMode,
> BwtConstruction<'a, 'b, 'r, I, O, B, P, A>
{
    /// By calling this function you are claiming that the frequency table is valid for the text
    /// for which this config is used later. Otherwise there is not guarantee for correct behavior
    /// of the C library.
    pub unsafe fn with_frequency_table(self, frequency_table: &'r mut [O]) -> Self {
        assert_eq!(frequency_table.len(), I::FREQUENCY_TABLE_SIZE);

        Self {
            frequency_table: Some(frequency_table),
            ..self
        }
    }
}

impl<'a, 'b, 'r, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    BwtConstruction<'a, 'b, 'r, I, O, B, P, NoAuxIndices>
{
    pub fn run(mut self) -> Result<Bwt<'b, I, B>, LibsaisError> {
        let text_len = self.text.as_ref().map_or_else(
            || self.bwt_buffer.as_ref().unwrap().len(),
            |text| text.len(),
        );

        let mut bwt = OwnedOrBorrowed::take_buffer_or_allocate(self.bwt_buffer.take(), || {
            vec![I::ZERO; text_len]
        });

        let mut temporary_array_buffer = if let Some(borrowed) = self.temporary_array_buffer.take()
        {
            Either::Right(borrowed)
        } else {
            Either::Left(suffix_array::allocate_suffix_array_buffer::<I, O>(
                self.extra_space_temporary_array_buffer,
                text_len,
            ))
        };
        if let Some(text) = self.text {
            suffix_array::sais_safety_checks(
                text,
                &temporary_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
            assert_eq!(text.len(), bwt.buffer.len());
        } else {
            suffix_array::sais_safety_checks(
                &bwt.buffer,
                &temporary_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
        }

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            suffix_array::cast_and_unpack_parameters(
                text_len,
                &temporary_array_buffer,
                self.thread_count,
                self.frequency_table.take(),
            );

        // this and the below break Rust's borrowing rules for bwt_buffer,
        // but the pointers are only used in the C code
        let text_ptr = self
            .text
            .map_or_else(|| bwt.buffer.as_ptr(), |text| text.as_ptr());

        // SAFETY:
        // buffer lens are safety checked (text, suffix array and bwt) with extra space in mind
        // suffix array buffer is at least as large as text, asserted in safety checks.
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size.
        // context must be of the correct type, because the API is typesafe and the parallelism decision was
        // forced to happen before the context was supplied.
        unsafe {
            SmallAlphabetFunctionsDispatch::<I, O, P>::libsais_bwt(
                text_ptr,
                bwt.buffer.as_mut_ptr(),
                temporary_array_buffer.as_mut_ptr(),
                text_len,
                extra_space,
                frequency_table_ptr,
                num_threads,
                self.context.map(|ctx| ctx.as_mut_ptr()),
            )
        }
        .into_primary_index_sais_result()
        .map(|primary_index| Bwt { bwt, primary_index })
    }
}

impl<
    'a,
    'b,
    'r,
    I: SmallAlphabet,
    O: OutputElement,
    BwtB: BufferMode,
    P: Parallelism,
    AuxB: BufferMode,
> BwtConstruction<'a, 'b, 'r, I, O, BwtB, P, AuxB>
{
    pub fn run(mut self) -> Result<BwtWithAuxIndices<'a, 'b, I, O, AuxB, BwtB>, LibsaisError> {
        let text_len = self.text.as_ref().map_or_else(
            || self.bwt_buffer.as_ref().unwrap().len(),
            |text| text.len(),
        );

        let mut bwt = OwnedOrBorrowed::take_buffer_or_allocate(self.bwt_buffer.take(), || {
            vec![I::ZERO; text_len]
        });

        let mut temporary_array_buffer = if let Some(borrowed) = self.temporary_array_buffer.take()
        {
            Either::Right(borrowed)
        } else {
            Either::Left(suffix_array::allocate_suffix_array_buffer::<I, O>(
                self.extra_space_temporary_array_buffer,
                text_len,
            ))
        };

        if let Some(text) = self.text {
            suffix_array::sais_safety_checks(
                text,
                &temporary_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
            assert_eq!(text.len(), bwt.buffer.len());
        } else {
            suffix_array::sais_safety_checks(
                &bwt.buffer,
                &temporary_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
        }

        let aux_indices_sampling_rate = self.aux_indices_sampling_rate.unwrap();
        let mut aux_indices =
            OwnedOrBorrowed::take_buffer_or_allocate(self.aux_indices_buffer.take(), || {
                vec![O::ZERO; aux_indices_sampling_rate.aux_indices_buffer_size(text_len)]
            });

        aux_indices_safety_checks_and_cast_sampling_rate(
            text_len,
            &aux_indices.buffer,
            aux_indices_sampling_rate,
        );

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            suffix_array::cast_and_unpack_parameters(
                text_len,
                &temporary_array_buffer,
                self.thread_count,
                self.frequency_table.take(),
            );

        // this and the below break Rust's borrowing rules for bwt_buffer,
        // but the pointers are only used in the C code
        let text_ptr = self
            .text
            .map_or_else(|| bwt.buffer.as_ptr(), |text| text.as_ptr());

        // SAFETY:
        // buffer lens are safety checked (text, suffix array, aux indices and bwt) with extra space, aux sampling rate in mind
        // suffix array buffer is at least as large as text, asserted in safety checks.
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size.
        // TODO context
        unsafe {
            SmallAlphabetFunctionsDispatch::<I, O, P>::libsais_bwt_aux(
                text_ptr,
                bwt.buffer.as_mut_ptr(),
                temporary_array_buffer.as_mut_ptr(),
                text_len,
                extra_space,
                frequency_table_ptr,
                aux_indices_sampling_rate.value,
                aux_indices.buffer.as_mut_ptr(),
                num_threads,
                self.context.map(|ctx| ctx.as_mut_ptr()),
            )
        }
        .into_empty_sais_result()
        .map(|_| BwtWithAuxIndices {
            bwt,
            aux_indices,
            aux_indices_sampling_rate,
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
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

#[derive(Debug, PartialEq, Eq, Hash)]
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
    pub(crate) aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
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

    pub fn aux_indices_sampling_rate(&self) -> AuxIndicesSamplingRate<O> {
        self.aux_indices_sampling_rate
    }

    pub fn into_parts(
        self,
    ) -> (
        BwtB::Buffer<'b, I>,
        AuxB::Buffer<'a, O>,
        AuxIndicesSamplingRate<O>,
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
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuxIndicesSamplingRate<O: OutputElementOrUndecided> {
    pub(crate) value: O,
    pub(crate) value_usize: usize,
}

impl<O: OutputElement> AuxIndicesSamplingRate<O> {
    pub fn value(&self) -> usize {
        self.value_usize
    }

    fn aux_indices_buffer_size(self, text_len: usize) -> usize {
        if text_len == 0 {
            0
        } else {
            (text_len - 1) / self.value_usize + 1
        }
    }
}

impl<O: OutputElement> From<O> for AuxIndicesSamplingRate<O> {
    fn from(value: O) -> Self {
        if value.into() < O::ZERO.into() {
            panic!("Aux indices sampling rate cannot be negative");
        }

        let value_usize = value.into() as usize;

        if value_usize < 2 {
            panic!("Aux indices sampling rate must be greater than 1");
        } else if value_usize.count_ones() != 1 {
            panic!("Aux indices sampling rate must be a power of two");
        } else {
            Self { value, value_usize }
        }
    }
}

pub(crate) trait IntoOtherInner<O2: OutputElement> {
    fn into_other_inner(self) -> AuxIndicesSamplingRate<O2>;
}

impl<O1: OutputElementOrUndecided, O2: OutputElement> IntoOtherInner<O2>
    for AuxIndicesSamplingRate<O1>
{
    fn into_other_inner(self) -> AuxIndicesSamplingRate<O2> {
        AuxIndicesSamplingRate {
            value: O2::try_from(self.value_usize)
                .expect("Auxiliary indices sampling rate needs to fit into output type"),
            value_usize: self.value_usize,
        }
    }
}

pub(crate) fn aux_indices_safety_checks_and_cast_sampling_rate<O: OutputElement>(
    text_len: usize,
    aux_indices_buffer: &[O],
    aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
) {
    assert_eq!(
        aux_indices_buffer.len(),
        aux_indices_sampling_rate.aux_indices_buffer_size(text_len)
    );
}
