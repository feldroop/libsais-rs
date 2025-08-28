/*!
 * Construct the [Burrows-Wheeler-Transform] (BWT) for a text using [`BwtConstruction`].
 *
 * To construct the BWT, a temporary (suffix) array is constructed by `libsais`. However,
 * this temporary array does not contain the suffix array after the procedure.
 *
 * The entry point to the API is the [`BwtConstruction`] builder-like struct. It is always required to
 * pass the input text, register the output element _of the temporary array_ and make a decision about
 * parallelisation. Further configuration options include supplying an output and/or temporary array buffer,
 * [`context`], metadata about the input text, usage of auxiliary indices, usage of additional memory
 * in the algorithm and instructing the library to replace the text by the BWT.
 *
 * The following is an example of the BWT construction that contains some of its unique configuration
 * options. See [`suffix_array`] for an example of most of the other options.
 *
 * ```
 * use libsais::{BwtConstruction, bwt, suffix_array::ExtraSpace};
 *
 * let mut text = vec![b'a'; 10];
 *
 * let res = BwtConstruction::replace_text(&mut text)
 *     .with_owned_temporary_array_buffer_and_extra_space32(ExtraSpace::Fixed { value: 10 })
 *     .single_threaded()
 *     .with_aux_indices(bwt::AuxIndicesSamplingRate::from(4))
 *     .run()
 *     .unwrap();
 *
 * println!("{:?}", res.bwt());
 * println!("{:?}", res.aux_indices());
 * ```
 *
 * # Output Conventions
 *
 * In the literature, the text input for the BWT construction is typically assumed to be terminated by a
 * unique, lexicographically smallest character. This character is called sentinel and denoted by $.
 *
 * `libsais` does not require the text to be terminated by a sentinel, but it behaves as if a sentinel
 * were present. In the output BWT, the sentinel is not included. Therefore, the BWT has the same length
 * as the input text.
 *
 * # Primary Index and Auxiliary Indices
 *
 * To recover the original text from a BWT, the primary index `i0` of the BWT is needed. It is defined as the
 * index for which `SUF[i0 - 1] = 0`, where `SUF` is the suffix array in `libsais` convention. The `-1`
 * is applied, because the suffix array also does not contain an entry for the sentinel. Such an entry
 * would always be at the first position.
 *
 * The primary index is part of the return type of [`BwtConstruction::run`]. It is 0 for the empty input text.
 *
 * `libsais` allows creating a subsampled array of auxiliary indices. These indices have the same role as the
 * primary index and are defined as follows: `AUX[i] == k => SUF[k - 1] = i * r`, where `r` is the sampling rate.
 * In particular, `AUX[0]` is th primary index. The auxiliary indices can be used in different ways.
 * For example, they allow significantly faster BWT reversal, both single and multi threaded. [Benchmark]
 *
 * # Return Type and Reversal
 *
 * The read-only return type of [`BwtConstruction::run`] bundles the BWT with either the primary index or the
 * auxiliary indices and their sampling rate. It is generic over whether an owned or borrowed output buffer is used.
 * The object can be destructured into parts or used to safely reverse the BWT and obtain the text again.
 *
 * An example of constructing and reversing the BWT can be found
 * [here](https://github.com/feldroop/libsais-rs/blob/master/examples/to_the_bwt_and_back.rs).
 *
 * [Burrows-Wheeler-Transform]: https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform
 * [`context`]: super::context
 * [Benchmark]: https://github.com/feldroop/benchmark_crates_io_sacas/blob/master/LIBSAIS_BWT_AUX.md
 */

use either::Either;

use std::marker::PhantomData;

use crate::{
    IntoSaisResult, LibsaisError, OutputElement, SmallAlphabet, ThreadCount,
    context::Context,
    generics_dispatch::{LibsaisFunctionsSmallAlphabet, SmallAlphabetFunctionsDispatch},
    owned_or_borrowed::OwnedOrBorrowed,
    suffix_array::{self, ExtraSpace},
    typestate::{
        AuxIndicesMode, BorrowedBuffer, BufferMode, BufferModeOrUndecided, NoAuxIndices,
        OutputElementOrUndecided, OwnedBuffer, Parallelism, ParallelismOrUndecided, SingleThreaded,
        Undecided,
    },
    unbwt::UnBwt,
};

#[cfg(feature = "openmp")]
use crate::typestate::MultiThreaded;

/// One of the two main entry points of this library, for constructing BWTs.
///
/// See [`bwt`](self) for details.
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

impl<'a, 'b, 'r, I: SmallAlphabet>
    BwtConstruction<'a, 'b, 'r, I, Undecided, Undecided, Undecided, NoAuxIndices>
{
    /// The first method to call if you don't want to replace the input text.
    ///
    /// The text has to be at most as long as the maximum value of the output element type
    /// you will choose for the temporary array.
    pub fn for_text(
        text: &'r [I],
    ) -> BwtConstruction<'a, 'b, 'r, I, Undecided, OwnedBuffer, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: Some(text),
            ..BwtConstruction::init()
        }
    }

    /// The first method to call if you want to replace the input text.
    ///
    /// The text has to be at most as long as the maximum value of the output element type
    /// you will choose for the temporary array.
    pub fn replace_text(
        text: &'b mut [I],
    ) -> BwtConstruction<'a, 'b, 'r, I, Undecided, BorrowedBuffer, Undecided, NoAuxIndices> {
        BwtConstruction {
            bwt_buffer: Some(text),
            ..BwtConstruction::init()
        }
    }
}

impl<'a, 'b, 'r, I: SmallAlphabet>
    BwtConstruction<'a, 'b, 'r, I, Undecided, OwnedBuffer, Undecided, NoAuxIndices>
{
    /// Optionally supply an output buffer for the result BWT.
    ///
    /// The buffer must have the same length as the text.
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
}

impl<'a, 'b, 'r, I: SmallAlphabet, B: BufferMode>
    BwtConstruction<'a, 'b, 'r, I, Undecided, B, Undecided, NoAuxIndices>
{
    /// Provide a buffer to the library in which the temporary array will be stored.
    ///
    /// The buffer has to be at least as large as the text, but at most as large as the maximum value
    /// of the output element type. Additional space might be used by the algorithm for better performance.
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

    /// Inform the library of your desired temporary array output element type,
    /// if you want the temporary array to be stored internally in a [`Vec`].
    pub fn with_owned_temporary_array_buffer<O: OutputElement>(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            ..BwtConstruction::init()
        }
    }

    /// Inform the library of your desired temporary array output element type,
    /// if you want the temporary array to be stored internally in a [`Vec<i32>`].
    pub fn with_owned_temporary_array_buffer32(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, i32, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            ..BwtConstruction::init()
        }
    }

    /// Inform the library of your desired temporary array output element type,
    /// if you want the temporary array to be stored internally in a [`Vec<i64>`].
    pub fn with_owned_temporary_array_buffer64(
        self,
    ) -> BwtConstruction<'a, 'b, 'r, I, i64, B, Undecided, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            ..BwtConstruction::init()
        }
    }

    /// Like [`Self::with_owned_temporary_array_buffer`], but with additional memory
    /// to be supplied to the algorithm, which might improve performance.
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

    /// Like [`Self::with_owned_temporary_array_buffer32`], but with additional memory
    /// to be supplied to the algorithm, which might improve performance.
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

    /// Like [`Self::with_owned_temporary_array_buffer64`], but with additional memory
    /// to be supplied to the algorithm, which might improve performance.
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

impl<'a, 'b, 'r, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    BwtConstruction<'a, 'b, 'r, I, O, B, P, NoAuxIndices>
{
    /// Instruct `libsais` to also generate auxiliary indicides.
    ///
    /// See [`bwt`](self#primary-index-and-auxiliary-indices) for details.
    pub fn with_aux_indices(
        mut self,
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> BwtConstruction<'a, 'b, 'r, I, O, B, P, OwnedBuffer> {
        self.aux_indices_sampling_rate = Some(aux_indices_sampling_rate);
        self.into_other_marker_type()
    }

    /// Instruct `libsais` to also generate auxiliary indicides in a borrowed buffer.
    ///
    /// The buffer must have exactly the following size: `(text_len - 1) / r + 1`,
    /// where `r` is the sampling rate.
    ///
    /// See [`bwt`](self#primary-index-and-auxiliary-indices) for details.
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
    ///
    /// Currently, this is only available for the `i32` output version. When using multiple threads,
    /// the thread count of the context must be equal to the threads count of this object.
    ///
    /// See [`context`](super::context) for further details.
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
    /// Supply the algorithm with a table that contains the number of occurences of each value.
    ///
    /// For `u8`-based texts, the table must have a size of 256, for `u16`-based texts, the table must have
    /// a size of 65536. This might slightly improve the performance of the algorithm.
    ///
    /// # Safety
    ///
    /// By calling this function you are claiming that the frequency table is valid for the text.
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

impl<'a, 'b, 'r, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    BwtConstruction<'a, 'b, 'r, I, O, B, P, NoAuxIndices>
{
    /// Construct the BWT for the given text.
    ///
    /// # Panics
    ///
    /// If any of the requirements of the methods called before are not met.
    ///
    /// # Returns
    ///
    /// An error or a type that bundles the BWT with the primary index.
    /// See [`bwt`](self#return-type-and-reversal) for details.
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
    /// Construct the BWT for the given text.
    ///
    /// # Panics
    ///
    /// If any of the requirements of the methods called before are not met.
    ///
    /// # Returns
    ///
    /// An error or a type that bundles the BWT with the auxiliary indices.
    /// See [`bwt`](self#return-type-and-reversal) for details.
    pub fn run(mut self) -> Result<BwtWithAuxIndices<'a, 'b, I, O, BwtB, AuxB>, LibsaisError> {
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

/// The read-only return type of a BWT construction without auxiliary indices.
///
/// It bundes the BWT and the primary index for safe BWT reversal.
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

    /// Construct this type without going through a [`BwtConstruction`] or by using the parts
    /// obtained by [`Self::into_parts`].
    ///
    /// # Safety
    ///
    /// You are claiming that the BWT  with the primary index is correct for some text according
    /// to the conventions of `libsais`.
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

/// The read-only return type of a BWT construction with auxiliary indices.
///
/// It bundes the BWT, auxiliary indices and sampling rate for safe BWT reversal.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BwtWithAuxIndices<
    'a,
    'b,
    I: SmallAlphabet,
    O: OutputElement,
    BwtB: BufferMode,
    AuxB: BufferMode,
> {
    pub(crate) bwt: OwnedOrBorrowed<'b, I, BwtB>,
    pub(crate) aux_indices: OwnedOrBorrowed<'a, O, AuxB>,
    pub(crate) aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
}

impl<'a, 'b, I: SmallAlphabet, O: OutputElement, BwtB: BufferMode, AuxB: BufferMode>
    BwtWithAuxIndices<'a, 'b, I, O, BwtB, AuxB>
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

    /// Construct this type without going through a [`BwtConstruction`] or by using the parts
    /// obtained by [`Self::into_parts`].
    ///
    /// # Safety
    ///
    /// You are claiming that the BWT and the auxiliary indices with the given sampling rate
    /// are correct for some text according to the conventions of `libsais`.
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

/// The sampling rate for auxiliary indices of the BWT construction
///
/// See [`bwt`](self#primary-index-and-auxiliary-indices) for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuxIndicesSamplingRate<O: OutputElementOrUndecided> {
    pub(crate) value: O,
    pub(crate) value_usize: usize,
}

impl<O: OutputElement> AuxIndicesSamplingRate<O> {
    /// Create a sampling rate from an output element type.
    ///
    /// # Panics
    ///
    /// The sampling rate must be a power of two and greater than 1. Otherwise this function panics.
    pub fn new(value: O) -> Self {
        Self::from(value)
    }

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
    /// Create a sampling rate from an output element type.
    ///
    /// # Panics
    ///
    /// The sampling rate must be a power of two and greater than 1. Otherwise this function panics.
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
