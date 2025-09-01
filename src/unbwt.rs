/*!
 * Recover the text from a Burrows-Wheeler-Transform.
 *
 * The [`UnBwt`] builder-like struct can only be obtained by running a BWT construction first or by
 * using an `unsafe` constructor of [`Bwt`](super::bwt::Bwt)/[`BwtWithAuxIndices`](super::bwt::BwtWithAuxIndices).
 *
 * For optimal performance, it is recommended to use auxiliary indices ([Benchmark]). Their usage is automatically
 * registered in the [`UnBwt`] builder depending on how you obtain it. Other configuration options include the
 * use of borrowed output and temporary array buffers, metadata about the BWT, usage of a [`context`](super::context)
 * and the possiblity of replacing the BWT by the resulting text.
 *
 * It is important to know that the temporary array buffer must have the size of the BWT _plus one_ in this operation.
 *
 * The following is a simple example of constructing and reversing a BWT:
 *
 *
 * ```
 * use libsais::BwtConstruction;
 *
 * let text = b"blablablabla".as_slice();
 *
 * let res = BwtConstruction::for_text(text)
 *     .with_owned_temporary_array_buffer32()
 *     .single_threaded()
 *     .run()
 *     .unwrap();
 *
 * println!("{:?}", res.bwt());
 *
 * let recovered_text = res
 *     .unbwt()
 *     .with_owned_temporary_array_buffer32()
 *     .single_threaded()
 *     .run()
 *     .unwrap();
 *
 * assert_eq!(text, recovered_text.as_slice());
 * ```
 *
 * A more elaborate example can be found [here].
 *
 * [here]: https://github.com/feldroop/libsais-rs/blob/master/examples/to_the_bwt_and_back.rs
 * [Benchmark]: https://github.com/feldroop/benchmark_crates_io_sacas/blob/master/LIBSAIS_BWT_AUX.md
 */

use either::Either;
use num_traits::NumCast;

use std::marker::PhantomData;

use crate::{
    InputElement, IntoSaisResult, LibsaisError, OutputElement, SmallAlphabet, ThreadCount,
    bwt::{AuxIndicesSamplingRate, IntoOtherInner},
    context::UnBwtContext,
    generics_dispatch::{LibsaisFunctionsSmallAlphabet, SmallAlphabetFunctionsDispatch},
    owned_or_borrowed::OwnedOrBorrowed,
    suffix_array,
    typestate::{
        BorrowedBuffer, BufferMode, OutputElementOrUndecided, OwnedBuffer, Parallelism,
        ParallelismOrUndecided, SingleThreaded, Undecided,
    },
};

#[cfg(feature = "openmp")]
use crate::typestate::MultiThreaded;

/// Recover the text from a BWT
///
/// See [`unbwt`](self) for details.
#[derive(Debug)]
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
    pub(crate) aux_indices_sampling_rate: Option<AuxIndicesSamplingRate<O>>,
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
            aux_indices_sampling_rate: self.aux_indices_sampling_rate.map(|r| r.into_other_inner()),
            aux_indices_buffer: None,
            _text_buffer_mode_marker: PhantomData,
        }
    }
}

// optional first or second choice: text
impl<'b, 'r, 't, I: SmallAlphabet, O: OutputElementOrUndecided, BwtB: BufferMode>
    UnBwt<'b, 'r, 't, I, O, BwtB, OwnedBuffer, Undecided>
{
    /// Optionally supply an output buffer for the result text.
    ///
    /// The buffer must have the same length as the BWT.
    pub fn in_borrowed_text_buffer(
        self,
        text: &'t mut [I],
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, BorrowedBuffer, Undecided> {
        self.into_other_marker_type_with_text(OwnedOrBorrowed::new(text))
    }

    /// Instruct the library to replace the input BWT by the recovered text.
    pub fn replace_bwt(mut self) -> UnBwt<'b, 'r, 'b, I, O, BwtB, BwtB, Undecided> {
        let bwt = self.bwt.take().unwrap();
        self.into_other_marker_type_with_text(bwt)
    }
}

// optional first or second choice: temporary array buffer type
impl<'b, 'r, 't, I: SmallAlphabet, BwtB: BufferMode, TextB: BufferMode>
    UnBwt<'b, 'r, 't, I, Undecided, BwtB, TextB, Undecided>
{
    /// Provide a buffer to the library in which the temporary array will be stored.
    ///
    /// The buffer must have the same as the input BWT _plus one_.
    pub fn with_borrowed_temporary_array_buffer<O: OutputElement>(
        self,
        temporary_array_buffer: &'r mut [O],
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(Some(temporary_array_buffer))
    }

    /// Inform the library of your desired temporary array output element type,
    /// if you want the temporary array to be stored internally in a [`Vec`].
    pub fn with_owned_temporary_array_buffer<O: OutputElement>(
        self,
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(None)
    }

    /// Inform the library of your desired temporary array output element type,
    /// if you want the temporary array to be stored internally in a [`Vec<i32>`].
    pub fn with_owned_temporary_array_buffer32(
        self,
    ) -> UnBwt<'b, 'r, 't, I, i32, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(None)
    }

    /// Inform the library of your desired temporary array output element type,
    /// if you want the temporary array to be stored internally in a [`Vec<i64>`].
    pub fn with_owned_temporary_array_buffer64(
        self,
    ) -> UnBwt<'b, 'r, 't, I, i64, BwtB, TextB, Undecided> {
        self.into_other_output_type_with_temporary_array_buffer(None)
    }
}

// if O is already known due to aux indices, this can be used
impl<'b, 'r, 't, I: SmallAlphabet, O: OutputElement, BwtB: BufferMode, TextB: BufferMode>
    UnBwt<'b, 'r, 't, I, O, BwtB, TextB, Undecided>
{
    /// Provide a buffer to the library in which the temporary array will be stored.
    ///
    /// The buffer must have the same as the input BWT _plus one_.
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
        // this is okay, because no context could have been supplied so far
        self.into_other_marker_type_without_context()
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        mut self,
        thread_count: ThreadCount,
    ) -> UnBwt<'b, 'r, 't, I, O, BwtB, TextB, MultiThreaded> {
        self.thread_count = thread_count;
        // this is okay, because no context could have been supplied so far
        self.into_other_marker_type_without_context()
    }
}

impl<'b, 'r, 't, I: SmallAlphabet, BwtB: BufferMode, TextB: BufferMode, P: Parallelism>
    UnBwt<'b, 'r, 't, I, i32, BwtB, TextB, P>
{
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    ///
    /// Currently, this is only available for the `i32` output version. When using multiple threads,
    /// the thread count of the context must be equal to the threads count of this object.
    ///
    /// See [`context`](super::context) for further details.
    pub fn with_context(self, context: &'r mut UnBwtContext<I, i32, P>) -> Self {
        Self {
            context: Some(context),
            ..self
        }
    }
}

impl<
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
    /// Supply the algorithm with a table that contains the number of occurences of each value.
    ///
    /// For `u8`-based BWTs, the table must have a size of 256, for `u16`-based BWTs, the table must have
    /// a size of 65536. This might slightly improve the performance of the algorithm.
    ///
    /// # Safety
    ///
    /// By calling this function you are claiming that the frequency table is valid for the BWT.
    ///
    /// # Panics
    ///
    /// If the frequency table has the wrong size.
    pub unsafe fn with_frequency_table(self, frequency_table: &'r mut [O]) -> Self {
        assert_eq!(frequency_table.len(), I::FREQUENCY_TABLE_SIZE);

        Self {
            frequency_table: Some(frequency_table),
            ..self
        }
    }
}

impl<
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
    /// Recover the original text for the given BWT.
    ///
    /// # Panics
    ///
    /// If any of the requirements of the methods called before are not met and if the input BWT is
    /// too large for the chosen output element type of the temporary array.
    ///
    /// # Returns
    ///
    /// An error or a simple wrapper type around the recovered text.
    pub fn run(mut self) -> Result<Text<'t, I, TextB>, LibsaisError> {
        let bwt_len = self.bwt.as_ref().map_or_else(
            || self.text.as_ref().unwrap().buffer.len(),
            |bwt| bwt.buffer.len(),
        );

        // if there is no text, TextB must be OwnedBuffer
        let mut text = self.text.take().unwrap_or_else(|| {
            OwnedOrBorrowed::take_buffer_or_allocate(None, || vec![I::zero(); bwt_len])
        });

        let mut temporary_array_buffer = if let Some(borrowed) = self.temporary_array_buffer.take()
        {
            Either::Right(borrowed)
        } else {
            Either::Left(vec![O::zero(); bwt_len + 1])
        };

        assert_eq!(bwt_len, text.buffer.len());
        assert_eq!(bwt_len + 1, temporary_array_buffer.len());
        assert!(temporary_array_buffer.len() <= <usize as NumCast>::from(O::max_value()).unwrap());

        if let Some(context) = self.context.as_ref() {
            assert_eq!(
                context.num_threads(),
                self.thread_count.value,
                "context needs to have the same number of threads as this config"
            );
        }

        let (_, bwt_len_output_type, num_threads, frequency_table_ptr) =
            suffix_array::cast_and_unpack_parameters(
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
        // context must be of the correct type, because the API is typesafe and the parallelism decision was
        // forced to happen before the context was supplied.
        if let Some(primary_index) = self.primary_index.take() {
            unsafe {
                SmallAlphabetFunctionsDispatch::<I, O, P>::libsais_unbwt(
                    bwt_ptr,
                    text.buffer.as_mut_ptr(),
                    temporary_array_buffer.as_mut_ptr(),
                    bwt_len_output_type,
                    frequency_table_ptr,
                    <O as NumCast>::from(primary_index)
                        .expect("primary index needs to fit into output type"),
                    num_threads,
                    self.context.map(|ctx| ctx.as_mut_ptr()),
                )
            }
        } else {
            let aux_indices_buffer = self.aux_indices_buffer.unwrap();
            let aux_indices_sampling_rate = self.aux_indices_sampling_rate.unwrap();

            crate::bwt::aux_indices_safety_checks_and_cast_sampling_rate(
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
                    aux_indices_sampling_rate.value,
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

/// A simple wrapper type around the recovered text.
#[derive(Debug, PartialEq, Eq, Hash)]
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
