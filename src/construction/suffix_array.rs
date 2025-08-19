use super::{AlphabetSize, ExtraSpace, IntoSaisResult, SaisError, ThreadCount};
use crate::context::SaisContext;
use crate::data_structures::{BorrowedSuffixArrayWithText, SuffixArrayWithText};
use crate::helpers;
use crate::type_model::*;

use std::marker::PhantomData;

pub struct SuffixArrayConstruction<
    't,
    'a,
    I: InputElement,
    O: OutputElementOrUndecided,
    B: BufferMode,
    P: Parallelism,
> {
    text: Option<&'t [I]>,
    text_mut: Option<&'t mut [I]>,
    suffix_array_buffer: Option<&'a mut [O]>,
    frequency_table: Option<&'a mut [O]>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    alphabet_size: AlphabetSize,
    extra_space: ExtraSpace,
    context: Option<&'a mut I::SingleThreadedContext>,
    _parallelism_marker: PhantomData<P>,
    _buffer_mode_marker: PhantomData<B>,
}

impl<'t, 'a, I: InputElement, O: OutputElementOrUndecided, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, B, P>
{
    fn init() -> Self {
        Self {
            text: None,
            text_mut: None,
            suffix_array_buffer: None,
            frequency_table: None,
            thread_count: ThreadCount::fixed(1),
            generalized_suffix_array: false,
            alphabet_size: AlphabetSize::ComputeFromMaxOfText,
            extra_space: ExtraSpace::Recommended,
            context: None,
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
        }
    }
}

impl<'t, 'a, I: InputElement, O: OutputElementOrUndecided, B1: BufferMode, P1: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, B1, P1>
{
    fn into_other_marker_type<B2: BufferMode, P2: Parallelism>(
        self,
    ) -> SuffixArrayConstruction<'t, 'a, I, O, B2, P2> {
        SuffixArrayConstruction {
            text: self.text,
            text_mut: self.text_mut,
            suffix_array_buffer: self.suffix_array_buffer,
            frequency_table: self.frequency_table,
            thread_count: self.thread_count,
            generalized_suffix_array: self.generalized_suffix_array,
            alphabet_size: self.alphabet_size,
            extra_space: self.extra_space,
            context: self.context,
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
        }
    }
}

// entry point to builder for small alphabets
impl<'t, 'a, I: SmallAlphabet>
    SuffixArrayConstruction<'t, 'a, I, Undecided, Undecided, SingleThreaded>
{
    pub fn for_text(text: &'t [I]) -> Self {
        Self {
            text: Some(text),
            ..Self::init()
        }
    }
}

// entry point to builder for large alphabets
impl<'t, 'a, I: LargeAlphabet>
    SuffixArrayConstruction<'t, 'a, I, Undecided, Undecided, SingleThreaded>
{
    pub fn for_text_mut(text: &'t mut [I]) -> Self {
        Self {
            text_mut: Some(text),
            ..Self::init()
        }
    }
}

// second choice: output type and buffer mode
impl<'t, 'a, I: InputElement>
    SuffixArrayConstruction<'t, 'a, I, Undecided, Undecided, SingleThreaded>
{
    pub fn in_borrowed_buffer<O>(
        self,
        suffix_array_buffer: &'a mut [O],
    ) -> SuffixArrayConstruction<'t, 'a, I, O, BorrowedBuffer, SingleThreaded>
    where
        O: OutputElement + IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            suffix_array_buffer: Some(suffix_array_buffer),
            text: self.text,
            text_mut: self.text_mut,
            thread_count: self.thread_count,
            ..SuffixArrayConstruction::init()
        }
    }

    pub fn in_owned_buffer<O>(
        self,
    ) -> SuffixArrayConstruction<'t, 'a, I, O, OwnedBuffer, SingleThreaded>
    where
        O: OutputElement + IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            text: self.text,
            text_mut: self.text_mut,
            thread_count: self.thread_count,
            ..SuffixArrayConstruction::init()
        }
    }
}

// optional choice at any time: threading
#[cfg(feature = "openmp")]
impl<'t, 'a, I: InputElement, O: OutputElement, B: BufferMode>
    SuffixArrayConstruction<'t, 'a, I, O, B, SingleThreaded>
{
    pub fn multi_threaded(
        mut self,
        thread_count: ThreadCount,
    ) -> SuffixArrayConstruction<'t, 'a, I, O, B, MultiThreaded> {
        self.thread_count = thread_count;
        self.into_other_marker_type()
    }
}

impl<'t, 'a, I: SmallAlphabet, B: BufferMode>
    SuffixArrayConstruction<'t, 'a, I, i32, B, SingleThreaded>
{
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded 32-bit output version.
    pub fn with_context(self, context: &'a mut I::SingleThreadedContext) -> Self {
        Self {
            context: Some(context),
            ..self
        }
    }
}

impl<'t, 'a, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, B, P>
{
    /// By calling this function you are claiming that the frequency table is valid for the text
    /// for which this config is used later. Otherwise there is not guarantee for correct behavior
    /// of the C library.
    pub unsafe fn with_frequency_table(self, frequency_table: &'a mut [O]) -> Self {
        assert_eq!(frequency_table.len(), I::FREQUENCY_TABLE_SIZE);

        Self {
            frequency_table: Some(frequency_table),
            ..self
        }
    }

    /// Construct the generalized suffix array, which is the suffix array of a set of strings.
    /// Conceptually, all suffixes of all of the strings will be sorted in a single array.
    /// The set of strings will be supplied to the algorithm by concatenating them separated by the 0 character
    /// (not ASCII '0'). The concatenated string additionally has to be terminated by a 0.
    pub fn generalized_suffix_array(self) -> Self {
        Self {
            generalized_suffix_array: true,
            ..self
        }
    }
}

impl<'t, 'a, I: LargeAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, B, P>
{
    /// By calling this function you are asserting that all values of the text
    /// you'll use for this configuration are in the range [0, alphabet_size)
    pub unsafe fn with_alphabet_size(self, alphabet_size: usize) -> Self {
        Self {
            alphabet_size: AlphabetSize::Fixed {
                value: alphabet_size,
            },
            ..self
        }
    }
}

impl<'t, 'a, I: InputElement, O: OutputElement, P: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, OwnedBuffer, P>
{
    pub fn with_extra_space_in_buffer(self, extra_space: ExtraSpace) -> Self {
        Self {
            extra_space,
            ..self
        }
    }
}

// -------------------- operations and runners for only suffix array and small alphabets --------------------
impl<'t, 'a, I: InputElement, O: OutputElement, P: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, OwnedBuffer, P>
{
    /// Construct the suffix array for the given text.
    pub fn construct(self) -> Result<SuffixArrayWithText<'t, I, O>, SaisError> {
        let text_len = self.text().len();

        let mut suffix_array_buffer =
            super::allocate_suffix_array_buffer::<I, O>(self.extra_space, text_len);

        let mut construction = self.into_other_marker_type::<BorrowedBuffer, P>();
        construction.suffix_array_buffer = Some(&mut suffix_array_buffer);

        let res = construction.construct_in_borrowed_buffer();

        match res {
            Ok(borrowed) => {
                let (_, text, is_generalized_suffix_array) = borrowed.into_parts();

                super::free_extra_space(&mut suffix_array_buffer, text_len);

                Ok(SuffixArrayWithText {
                    suffix_array: suffix_array_buffer,
                    text,
                    is_generalized_suffix_array,
                })
            }
            Err(e) => Err(e),
        }
    }
}

impl<'t, 'a, I: InputElement, O: OutputElement, P: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, BorrowedBuffer, P>
{
    /// Construct the suffix array for the given text in the provided buffer.
    /// The buffer must be at least as large as the text. Additional space at the end
    /// will be used as extra space.
    /// The suffix array will be written to the start of the buffer
    pub fn construct_in_borrowed_buffer(
        mut self,
    ) -> Result<BorrowedSuffixArrayWithText<'a, 't, I, O>, SaisError> {
        let suffix_array_buffer = self.suffix_array_buffer.take().unwrap();

        if self.text().is_empty() {
            return Ok(BorrowedSuffixArrayWithText {
                suffix_array_buffer,
                text: self.take_text(),
                is_generalized_suffix_array: self.generalized_suffix_array,
            });
        }

        super::sais_safety_checks(
            self.text(),
            suffix_array_buffer,
            &self.context,
            self.thread_count,
            self.generalized_suffix_array,
        );

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            super::cast_and_unpack_parameters(
                self.text().len(),
                suffix_array_buffer,
                self.thread_count,
                self.frequency_table.take(),
            );

        // SAFETY:
        // text len is asserted to be in required range in safety checks.
        // suffix array buffer is at least as large as text, asserted in safety checks.
        // if there is a context it has the correct type, because that was claimed in an unsafe impl
        // for InputElementDecided.
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size (only small alphabets).
        // alphabet size was either set as the max of the text + 1 or claimed to be
        // correct in an unsafe function (only large alphabets).
        if let Some(text) = self.text {
            // small alphabets
            unsafe {
                <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::SmallAlphabetFunctions::run_libsais(
                    text.as_ptr(),
                    suffix_array_buffer.as_mut_ptr(),
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                    num_threads,
                    self.generalized_suffix_array,
                    self.context.take().map(|ctx| ctx.as_mut_ptr()),
                )
            }
        } else {
            // large alphabets
            let text = self.text_mut.as_mut().expect("A text shared or mutable reference must have been given in the builder process");

            let alphabet_size: O = match self.alphabet_size {
                AlphabetSize::ComputeFromMaxOfText => {
                    helpers::compute_and_validate_alphabet_size(text).unwrap_or_else(|e| panic!("{e}"))
                }
                AlphabetSize::Fixed { value } => value
                    .try_into()
                    .expect("The alphabet size needs to fit into the output type."),
            };

            unsafe {
                <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::LargeAlphabetFunctions::run_libsais_large_alphabet(
                    (*text).as_mut_ptr(),
                    suffix_array_buffer.as_mut_ptr(),
                    text_len,
                    alphabet_size,
                    extra_space,
                    num_threads,
                )
            }
        }.into_empty_sais_result().map(|_| BorrowedSuffixArrayWithText {
            suffix_array_buffer: &suffix_array_buffer[..self.text().len()],
            text: self.take_text(),
            is_generalized_suffix_array: self.generalized_suffix_array
        })
    }
}

impl<'t, 'a, I: InputElement, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'t, 'a, I, O, B, P>
{
    fn text(&self) -> &[I] {
        self.text.unwrap_or_else(|| self.text_mut.as_ref().unwrap())
    }

    fn take_text(&mut self) -> &'t [I] {
        self.text
            .take()
            .unwrap_or_else(|| self.text_mut.take().unwrap())
    }
}
