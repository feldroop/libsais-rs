use super::{AlphabetSize, ExtraSpace, IntoSaisResult, SaisError, ThreadCount};
use crate::context::SaisContext;
use crate::data_structures::{OwnedOrBorrowed, SuffixArrayWithText};
use crate::helpers;
use crate::type_model::*;

use std::marker::PhantomData;

pub struct SuffixArrayConstruction<
    'r,
    's,
    't,
    I: InputElement,
    O: OutputElementOrUndecided,
    B: BufferModeOrUndecided,
    P: Parallelism,
> {
    text: Option<&'t [I]>,
    text_mut: Option<&'t mut [I]>,
    suffix_array_buffer: Option<&'s mut [O]>,
    frequency_table: Option<&'r mut [O]>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    alphabet_size: AlphabetSize,
    extra_space: ExtraSpace,
    context: Option<&'r mut I::SingleThreadedContext>,
    _parallelism_marker: PhantomData<P>,
    _buffer_mode_marker: PhantomData<B>,
}

impl<
    'r,
    's,
    't,
    I: InputElement,
    O: OutputElementOrUndecided,
    B: BufferModeOrUndecided,
    P: Parallelism,
> SuffixArrayConstruction<'r, 's, 't, I, O, B, P>
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

// entry point to builder for small alphabets
impl<'t, I: SmallAlphabet>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, SingleThreaded>
{
    pub fn for_text(text: &'t [I]) -> Self {
        Self {
            text: Some(text),
            ..Self::init()
        }
    }
}

// entry point to builder for large alphabets
impl<'t, I: LargeAlphabet>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, SingleThreaded>
{
    pub fn for_text_mut(text: &'t mut [I]) -> Self {
        Self {
            text_mut: Some(text),
            ..Self::init()
        }
    }
}

// second choice: output type and buffer mode
impl<'t, I: InputElement>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, SingleThreaded>
{
    pub fn in_borrowed_buffer<'s, O>(
        self,
        suffix_array_buffer: &'s mut [O],
    ) -> SuffixArrayConstruction<'static, 's, 't, I, O, BorrowedBuffer, SingleThreaded>
    where
        O: IsValidOutputFor<I>,
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
    ) -> SuffixArrayConstruction<'static, 'static, 't, I, O, OwnedBuffer, SingleThreaded>
    where
        O: IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            text: self.text,
            text_mut: self.text_mut,
            thread_count: self.thread_count,
            ..SuffixArrayConstruction::init()
        }
    }

    pub fn in_owned_buffer32(
        self,
    ) -> SuffixArrayConstruction<'static, 'static, 't, I, i32, OwnedBuffer, SingleThreaded>
    where
        i32: IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            text: self.text,
            text_mut: self.text_mut,
            thread_count: self.thread_count,
            ..SuffixArrayConstruction::init()
        }
    }

    pub fn in_owned_buffer64(
        self,
    ) -> SuffixArrayConstruction<'static, 'static, 't, I, i64, OwnedBuffer, SingleThreaded>
    where
        i64: IsValidOutputFor<I>,
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
impl<'r, 's, 't, I: InputElement, O: OutputElement, B: BufferMode>
    SuffixArrayConstruction<'r, 's, 't, I, O, B, SingleThreaded>
{
    pub fn multi_threaded(
        self,
        thread_count: ThreadCount,
    ) -> SuffixArrayConstruction<'r, 's, 't, I, O, B, MultiThreaded> {
        SuffixArrayConstruction {
            text: self.text,
            text_mut: self.text_mut,
            suffix_array_buffer: self.suffix_array_buffer,
            frequency_table: self.frequency_table,
            thread_count,
            generalized_suffix_array: self.generalized_suffix_array,
            alphabet_size: self.alphabet_size,
            extra_space: self.extra_space,
            context: self.context,
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
        }
    }
}

impl<'r, 's, 't, I: SmallAlphabet, B: BufferMode>
    SuffixArrayConstruction<'r, 's, 't, I, i32, B, SingleThreaded>
{
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded 32-bit output version.
    pub fn with_context(self, context: &'r mut I::SingleThreadedContext) -> Self {
        Self {
            context: Some(context),
            ..self
        }
    }
}

impl<'r, 's, 't, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'r, 's, 't, I, O, B, P>
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

impl<'r, 's, 't, I: LargeAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'r, 's, 't, I, O, B, P>
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

impl<'r, 's, 't, I: InputElement, O: OutputElement, P: Parallelism>
    SuffixArrayConstruction<'r, 's, 't, I, O, OwnedBuffer, P>
{
    pub fn with_extra_space_in_buffer(self, extra_space: ExtraSpace) -> Self {
        Self {
            extra_space,
            ..self
        }
    }
}

// -------------------- operations and runners for only suffix array and small alphabets --------------------
impl<'r, 's, 't, I: InputElement, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'r, 's, 't, I, O, B, P>
{
    /// Construct the suffix array for the given text.
    pub fn run(mut self) -> Result<SuffixArrayWithText<'s, 't, I, O, B>, SaisError> {
        let text_len = self.text().len();
        let mut suffix_array =
            OwnedOrBorrowed::take_buffer_or_allocate(self.suffix_array_buffer.take(), || {
                super::allocate_suffix_array_buffer::<I, O>(self.extra_space, text_len)
            });

        let res = self.construct_in_buffer(&mut suffix_array.buffer);

        suffix_array.shorten_buffer_to(text_len);

        res.map(|_| SuffixArrayWithText {
            suffix_array,
            text: self.take_text(),
            is_generalized_suffix_array: self.generalized_suffix_array,
        })
    }

    fn construct_in_buffer(&mut self, suffix_array_buffer: &mut [O]) -> Result<(), SaisError> {
        if self.text().is_empty() {
            return Ok(());
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
                <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::SmallAlphabetFunctions::libsais(
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
                <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::LargeAlphabetFunctions::libsais_large_alphabet(
                    (*text).as_mut_ptr(),
                    suffix_array_buffer.as_mut_ptr(),
                    text_len,
                    alphabet_size,
                    extra_space,
                    num_threads,
                )
            }
        }.into_empty_sais_result()
    }

    fn text(&self) -> &[I] {
        self.text.unwrap_or_else(|| self.text_mut.as_ref().unwrap())
    }

    fn take_text(&mut self) -> &'t [I] {
        self.text
            .take()
            .unwrap_or_else(|| self.text_mut.take().unwrap())
    }
}
