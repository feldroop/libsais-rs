use std::marker::PhantomData;

use either::Either;

use super::{IntoSaisResult, LibsaisError};
use crate::{
    ThreadCount,
    construction::AuxIndicesSamplingRate,
    context::UnBwtContext,
    data_structures::{OwnedOrBorrowed, Text},
    type_model::{
        BorrowedBuffer, BufferMode, LibsaisFunctionsSmallAlphabet, OutputElement,
        OutputElementOrUndecided, OwnedBuffer, Parallelism, ParallelismOrUndecided, SingleThreaded,
        SmallAlphabet, SmallAlphabetFunctionsDispatch, Undecided,
    },
};

#[cfg(feature = "openmp")]
use crate::type_model::MultiThreaded;

pub struct UnBwt<
    'b,
    'r,
    't,
    I: SmallAlphabet,
    O: OutputElementOrUndecided,
    BwtB: BufferMode,
    TextB: BufferMode,
    P: ParallelismOrUndecided,
> {
    pub(crate) bwt: Option<OwnedOrBorrowed<'b, I, BwtB>>,
    pub(crate) text: Option<OwnedOrBorrowed<'t, I, TextB>>,
    pub(crate) temporary_array_buffer: Option<&'r mut [O]>,
    pub(crate) frequency_table: Option<&'r mut [O]>,
    pub(crate) thread_count: ThreadCount,
    pub(crate) context: Option<&'r mut UnBwtContext<I, O, P>>,
    pub(crate) primary_index: Option<usize>,
    pub(crate) aux_indices_sampling_rate: Option<AuxIndicesSamplingRate>,
    pub(crate) aux_indices_buffer: Option<Either<Vec<O>, &'r mut [O]>>,
    pub(crate) _text_buffer_mode_marker: PhantomData<TextB>,
}

impl<
    'b,
    'r,
    't1,
    I: SmallAlphabet,
    O: OutputElementOrUndecided,
    BwtB: BufferMode,
    TextB1: BufferMode,
    P1: ParallelismOrUndecided,
> UnBwt<'b, 'r, 't1, I, O, BwtB, TextB1, P1>
{
    fn into_other_marker_type_with_text<'t2, TextB2: BufferMode>(
        self,
        text_buffer: OwnedOrBorrowed<'t2, I, TextB2>,
    ) -> UnBwt<'b, 'r, 't2, I, O, BwtB, TextB2, P1> {
        UnBwt {
            bwt: self.bwt,
            text: Some(text_buffer),
            temporary_array_buffer: self.temporary_array_buffer,
            frequency_table: self.frequency_table,
            thread_count: self.thread_count,
            context: self.context,
            primary_index: self.primary_index,
            aux_indices_sampling_rate: self.aux_indices_sampling_rate,
            aux_indices_buffer: self.aux_indices_buffer,
            _text_buffer_mode_marker: PhantomData,
        }
    }

    fn into_other_marker_type_without_context<P2: Parallelism>(
        self,
    ) -> UnBwt<'b, 'r, 't1, I, O, BwtB, TextB1, P2> {
        UnBwt {
            bwt: self.bwt,
            text: self.text,
            temporary_array_buffer: self.temporary_array_buffer,
            frequency_table: self.frequency_table,
            thread_count: self.thread_count,
            context: None,
            primary_index: self.primary_index,
            aux_indices_sampling_rate: self.aux_indices_sampling_rate,
            aux_indices_buffer: self.aux_indices_buffer,
            _text_buffer_mode_marker: PhantomData,
        }
    }
}

impl<
    'b,
    'r,
    't,
    I: SmallAlphabet,
    O1: OutputElementOrUndecided,
    BwtB: BufferMode,
    TextB: BufferMode,
> UnBwt<'b, 'r, 't, I, O1, BwtB, TextB, Undecided>
{
    fn into_other_output_type_with_temporary_array_buffer<O2: OutputElement>(
        self,
        temporary_array_buffer: Option<&'r mut [O2]>,
    ) -> UnBwt<'b, 'r, 't, I, O2, BwtB, TextB, Undecided> {
        UnBwt {
            bwt: self.bwt,
            text: self.text,
            temporary_array_buffer,
            frequency_table: None,
            thread_count: self.thread_count,
            context: None,
            primary_index: self.primary_index,
            aux_indices_sampling_rate: self.aux_indices_sampling_rate,
            aux_indices_buffer: None,
            _text_buffer_mode_marker: PhantomData,
        }
    }
}

// optional first or second choice: text
impl<'b, 'r, 't, I: SmallAlphabet, O: OutputElementOrUndecided, BwtB: BufferMode>
    UnBwt<'b, 'r, 't, I, O, BwtB, OwnedBuffer, Undecided>
{
    pub fn in_borrowed_text_buffer(
        self,
        text: &'t mut [I],
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, BorrowedBuffer, Undecided> {
        self.into_other_marker_type_with_text(OwnedOrBorrowed::new(text))
    }

    pub fn replace_bwt(mut self) -> UnBwt<'b, 'r, 'b, I, O, BwtB, BwtB, Undecided> {
        let bwt = self.bwt.take().unwrap();
        self.into_other_marker_type_with_text(bwt)
    }
}

// optional first or second choice: temporary array buffer type
impl<'b, 'r, 't, I: SmallAlphabet, BwtB: BufferMode, TextB: BufferMode>
    UnBwt<'b, 'r, 't, I, Undecided, BwtB, TextB, Undecided>
{
    pub fn with_borrowed_temporary_array_buffer<O: OutputElement>(
        self,
        temporary_array_buffer: &'r mut [O],
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(Some(temporary_array_buffer))
    }

    pub fn with_owned_temporary_array_buffer<O: OutputElement>(
        self,
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(None)
    }

    pub fn with_owned_temporary_array_buffer32<O: OutputElement>(
        self,
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(None)
    }

    pub fn with_owned_temporary_array_buffer64<O: OutputElement>(
        self,
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(None)
    }
}

// if O is already known due to aux indices, this can be used
impl<'b, 'r, 't, I: SmallAlphabet, O: OutputElement, BwtB: BufferMode, TextB: BufferMode>
    UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided>
{
    pub fn with_borrowed_temporary_array_buffer(
        self,
        temporary_array_buffer: &'r mut [O],
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(Some(temporary_array_buffer))
    }
}

impl<'b, 'r, 't, I: SmallAlphabet, O: OutputElement, BwtB: BufferMode, TextB: BufferMode>
    UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided>
{
    pub fn single_threaded(self) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, SingleThreaded> {
        self.into_other_marker_type_without_context()
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        mut self,
        thread_count: ThreadCount,
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, MultiThreaded> {
        self.thread_count = thread_count;
        self.into_other_marker_type_without_context()
    }
}

impl<'b, 'r, 't, I: SmallAlphabet, BwtB: BufferMode, TextB: BufferMode>
    UnBwt<'b, 'r, 't, I, i32, BwtB, TextB, SingleThreaded>
{
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded 32-bit output version.
    pub fn with_context(self, context: &'r mut UnBwtContext<I, i32, SingleThreaded>) -> Self {
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
    't,
    I: SmallAlphabet,
    O: OutputElement,
    BwtB: BufferMode,
    TextB: BufferMode,
    P: Parallelism,
> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, P>
{
    /// By calling this function you are claiming that the frequency table is valid for the bwt/text
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

impl<
    'a,
    'b,
    'r,
    't,
    I: SmallAlphabet,
    O: OutputElement,
    BwtB: BufferMode,
    TextB: BufferMode,
    P: Parallelism,
> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, P>
{
    pub fn run(mut self) -> Result<Text<'t, I, TextB>, LibsaisError> {
        let bwt_len = self.bwt.as_ref().map_or_else(
            || self.text.as_ref().unwrap().buffer.len(),
            |bwt| bwt.buffer.len(),
        );

        // if there is no text, TextB must be OwnedBuffer
        let mut text = self.text.take().unwrap_or_else(|| {
            OwnedOrBorrowed::take_buffer_or_allocate(None, || vec![I::ZERO; bwt_len])
        });

        let mut temporary_array_buffer = if let Some(borrowed) = self.temporary_array_buffer.take()
        {
            Either::Right(borrowed)
        } else {
            Either::Left(vec![O::ZERO; bwt_len + 1])
        };

        assert_eq!(bwt_len, text.buffer.len());
        assert_eq!(bwt_len + 1, temporary_array_buffer.len());
        assert!(temporary_array_buffer.len() <= O::MAX.into() as usize);

        if let Some(context) = self.context.as_ref() {
            assert_eq!(
                context.num_threads(),
                self.thread_count.value,
                "context needs to have the same number of threads as this config"
            );
        }

        let (_, bwt_len_output_type, num_threads, frequency_table_ptr) =
            super::cast_and_unpack_parameters(
                bwt_len,
                &temporary_array_buffer,
                self.thread_count,
                self.frequency_table.take(),
            );

        // this and the below break Rust's borrowing rules for bwt_buffer,
        // but the pointers are only used in the C code
        let bwt_ptr = self
            .bwt
            .as_ref()
            .map_or_else(|| text.buffer.as_ptr(), |bwt| bwt.buffer.as_ptr());

        // SAFETY:
        // bwt temporary array and text len are asserted to be correct.
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size.
        // primary index/aux indices must be correct, because they were attained either from a BwtConstruction
        // or claimed to be correct in an unsafe function.
        // TODO context
        if let Some(primary_index) = self.primary_index.take() {
            unsafe {
                SmallAlphabetFunctionsDispatch::<I, O, P>::libsais_unbwt(
                    bwt_ptr,
                    text.buffer.as_mut_ptr(),
                    temporary_array_buffer.as_mut_ptr(),
                    bwt_len_output_type,
                    frequency_table_ptr,
                    O::try_from(primary_index)
                        .expect("primary index needs to fit into output type"),
                    num_threads,
                    self.context.map(|ctx| ctx.as_mut_ptr()),
                )
            }
        } else {
            let aux_indices_buffer = self.aux_indices_buffer.unwrap();
            let aux_indices_sampling_rate = self.aux_indices_sampling_rate.unwrap();

            let aux_indices_sampling_rate_output_type =
                super::aux_indices_safety_checks_and_cast_sampling_rate(
                    bwt_len,
                    &aux_indices_buffer,
                    aux_indices_sampling_rate,
                );

            unsafe {
                SmallAlphabetFunctionsDispatch::<I, O, P>::libsais_unbwt_aux(
                    bwt_ptr,
                    text.buffer.as_mut_ptr(),
                    temporary_array_buffer.as_mut_ptr(),
                    bwt_len_output_type,
                    frequency_table_ptr,
                    aux_indices_sampling_rate_output_type,
                    aux_indices_buffer.as_ptr(),
                    num_threads,
                    self.context.map(|ctx| ctx.as_mut_ptr()),
                )
            }
        }
        .into_empty_sais_result()
        .map(|_| Text { text })
    }
}
