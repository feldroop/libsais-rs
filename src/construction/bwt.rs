use either::Either;

use super::{AuxIndicesSamplingRate, ExtraSpace, IntoSaisResult, SaisError, ThreadCount};
use crate::context::Context;
use crate::data_structures::{Bwt, BwtWithAuxIndices, OwnedOrBorrowed};
use crate::type_model::*;

use std::marker::PhantomData;

pub struct BwtConstruction<
    'a,
    'b,
    'r,
    I: InputElement,
    O: OutputElementOrUndecided,
    B: BufferModeOrUndecided,
    P: ParallelismOrUndecided,
    A: AuxIndicesMode,
> {
    text: Option<&'r [I]>,
    bwt_buffer: Option<&'b mut [I]>,
    temporary_suffix_array_buffer: Option<&'r mut [O]>,
    frequency_table: Option<&'r mut [O]>,
    extra_space_temporary_suffix_array_buffer: ExtraSpace,
    thread_count: ThreadCount,
    context: Option<&'r mut Context<I, O, P>>,
    aux_indices_sampling_rate: Option<AuxIndicesSamplingRate>,
    aux_indices_buffer: Option<&'a mut [O]>,
    _parallelism_marker: PhantomData<P>,
    _buffer_mode_marker: PhantomData<B>,
    _aux_indices_mode_marker: PhantomData<A>,
}

impl<
    'a,
    'b,
    'r,
    I: InputElement,
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
            temporary_suffix_array_buffer: None,
            frequency_table: None,
            thread_count: ThreadCount::fixed(1),
            extra_space_temporary_suffix_array_buffer: ExtraSpace::Recommended,
            context: None,
            aux_indices_sampling_rate: None,
            aux_indices_buffer: None,
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
            _aux_indices_mode_marker: PhantomData,
        }
    }
}

impl<
    'a,
    'b,
    'r,
    I: InputElement,
    O: OutputElementOrUndecided,
    B1: BufferModeOrUndecided,
    P1: ParallelismOrUndecided,
    A1: AuxIndicesMode,
> BwtConstruction<'a, 'b, 'r, I, O, B1, P1, A1>
{
    fn into_other_marker_type<
        B2: BufferModeOrUndecided,
        P2: ParallelismOrUndecided,
        A2: AuxIndicesMode,
    >(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B2, P2, A2> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_suffix_array_buffer: self.temporary_suffix_array_buffer,
            frequency_table: self.frequency_table,
            thread_count: self.thread_count,
            extra_space_temporary_suffix_array_buffer: self
                .extra_space_temporary_suffix_array_buffer,
            context: None, // TODO
            aux_indices_sampling_rate: self.aux_indices_sampling_rate,
            aux_indices_buffer: self.aux_indices_buffer,
            _parallelism_marker: PhantomData,
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
            thread_count: self.thread_count,
            ..BwtConstruction::init()
        }
    }

    pub fn in_owned_buffer(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, Undecided, OwnedBuffer, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            thread_count: self.thread_count,
            ..BwtConstruction::init()
        }
    }
}

// second choice: output type
impl<'a, 'b, 'r, I: SmallAlphabet, B: BufferMode>
    BwtConstruction<'a, 'b, 'r, I, Undecided, B, Undecided, NoAuxIndices>
{
    pub fn with_borrowed_temporary_suffix_array_buffer<O: OutputElement>(
        self,
        temporary_suffix_array_buffer: &'r mut [O],
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_suffix_array_buffer: Some(temporary_suffix_array_buffer),
            thread_count: self.thread_count,
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_suffix_array_buffer<O: OutputElement>(
        self,
        extra_space: ExtraSpace,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            thread_count: self.thread_count,
            extra_space_temporary_suffix_array_buffer: extra_space,
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
        self.into_other_marker_type()
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        mut self,
        thread_count: ThreadCount,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, MultiThreaded, NoAuxIndices> {
        self.thread_count = thread_count;
        self.into_other_marker_type()
    }
}

// optional choice at any time: with auxiliary indices
impl<'a, 'b, 'r, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    BwtConstruction<'a, 'b, 'r, I, O, B, P, NoAuxIndices>
{
    pub fn with_aux_indices(
        mut self,
        aux_indices_sampling_rate: AuxIndicesSamplingRate,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, P, OwnedBuffer> {
        self.aux_indices_sampling_rate = Some(aux_indices_sampling_rate);
        self.into_other_marker_type()
    }

    pub fn with_aux_indices_in_buffer(
        mut self,
        aux_indices_sampling_rate: AuxIndicesSamplingRate,
        aux_indices_buffer: &'a mut [O],
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, P, BorrowedBuffer> {
        self.aux_indices_sampling_rate = Some(aux_indices_sampling_rate);
        self.aux_indices_buffer = Some(aux_indices_buffer);
        self.into_other_marker_type()
    }
}

impl<'a, 'b, 'r, I: SmallAlphabet, B: BufferMode, A: AuxIndicesMode>
    BwtConstruction<'a, 'b, 'r, I, i32, B, SingleThreaded, A>
{
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded 32-bit output version.
    pub fn with_context(self, context: &'r mut Context<I, i32, SingleThreaded>) -> Self {
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
    pub fn run(mut self) -> Result<Bwt<'b, I, B>, SaisError> {
        let text_len = self.text.as_ref().map_or_else(
            || self.bwt_buffer.as_ref().unwrap().len(),
            |text| text.len(),
        );

        let mut bwt = OwnedOrBorrowed::take_buffer_or_allocate(self.bwt_buffer.take(), || {
            vec![I::ZERO; text_len]
        });

        let mut suffix_array_buffer =
            if let Some(borrowed) = self.temporary_suffix_array_buffer.take() {
                Either::Right(borrowed)
            } else {
                Either::Left(super::allocate_suffix_array_buffer::<I, O>(
                    self.extra_space_temporary_suffix_array_buffer,
                    text_len,
                ))
            };
        if let Some(text) = self.text {
            super::sais_safety_checks(
                text,
                &suffix_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
            assert_eq!(text.len(), bwt.buffer.len());
        } else {
            super::sais_safety_checks(
                &bwt.buffer,
                &suffix_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
        }

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            super::cast_and_unpack_parameters(
                text_len,
                &suffix_array_buffer,
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
        // TODO context
        unsafe {
            SmallAlphabetFunctionsDispatch::<I, O, P>::libsais_bwt(
                text_ptr,
                bwt.buffer.as_mut_ptr(),
                suffix_array_buffer.as_mut_ptr(),
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
    pub fn run(mut self) -> Result<BwtWithAuxIndices<'a, 'b, I, O, AuxB, BwtB>, SaisError> {
        let text_len = self.text.as_ref().map_or_else(
            || self.bwt_buffer.as_ref().unwrap().len(),
            |text| text.len(),
        );

        let mut bwt = OwnedOrBorrowed::take_buffer_or_allocate(self.bwt_buffer.take(), || {
            vec![I::ZERO; text_len]
        });

        let mut suffix_array_buffer =
            if let Some(borrowed) = self.temporary_suffix_array_buffer.take() {
                Either::Right(borrowed)
            } else {
                Either::Left(super::allocate_suffix_array_buffer::<I, O>(
                    self.extra_space_temporary_suffix_array_buffer,
                    text_len,
                ))
            };
        if let Some(text) = self.text {
            super::sais_safety_checks(
                text,
                &suffix_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
            assert_eq!(text.len(), bwt.buffer.len());
        } else {
            super::sais_safety_checks(
                &bwt.buffer,
                &suffix_array_buffer,
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
        let aux_indices_sampling_rate_output_type =
            super::aux_indices_safety_checks_and_cast_sampling_rate(
                text_len,
                &aux_indices.buffer,
                aux_indices_sampling_rate,
            );

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            super::cast_and_unpack_parameters(
                text_len,
                &suffix_array_buffer,
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
                suffix_array_buffer.as_mut_ptr(),
                text_len,
                extra_space,
                frequency_table_ptr,
                aux_indices_sampling_rate_output_type,
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
