/*!
 * Construct (generalized) suffix array (GSA) for a text using [SuffixArrayConstruction].
 */

use either::Either;

use crate::{
    InputElement, IntoSaisResult, IsValidOutputFor, LargeAlphabet, LibsaisError, OutputElement,
    SmallAlphabet, SupportsPlcpOutputFor, ThreadCount,
    context::Context,
    generics_dispatch::{
        LargeAlphabetFunctionsDispatch, LibsaisFunctionsLargeAlphabet,
        LibsaisFunctionsSmallAlphabet, SmallAlphabetFunctionsDispatch,
    },
    owned_or_borrowed::OwnedOrBorrowed,
    plcp::PlcpConstruction,
    type_state::{
        BorrowedBuffer, BufferMode, BufferModeOrUndecided, OutputElementOrUndecided, OwnedBuffer,
        Parallelism, ParallelismOrUndecided, SingleThreaded, Undecided,
    },
};

#[cfg(feature = "openmp")]
use crate::type_state::MultiThreaded;

use std::marker::PhantomData;

/// Test
#[derive(Debug)]
pub struct SuffixArrayConstruction<
    'r,
    's,
    't,
    I: InputElement,
    O: OutputElementOrUndecided,
    B: BufferModeOrUndecided,
    P: ParallelismOrUndecided,
> {
    text: Option<Either<&'t [I], &'t mut [I]>>,
    suffix_array_buffer: Option<&'s mut [O]>,
    frequency_table: Option<&'r mut [O]>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    alphabet_size: AlphabetSizeInner<O>,
    extra_space: ExtraSpace,
    context: Option<&'r mut Context<I, O, P>>,
    _buffer_mode_marker: PhantomData<B>,
}

impl<
    'r,
    's,
    't,
    I: InputElement,
    O: OutputElementOrUndecided,
    B: BufferModeOrUndecided,
    P: ParallelismOrUndecided,
> SuffixArrayConstruction<'r, 's, 't, I, O, B, P>
{
    fn init() -> Self {
        Self {
            text: None,
            suffix_array_buffer: None,
            frequency_table: None,
            thread_count: ThreadCount::fixed(1),
            generalized_suffix_array: false,
            alphabet_size: AlphabetSizeInner::ComputeFromMaxOfText,
            extra_space: ExtraSpace::Recommended,
            context: None,
            _buffer_mode_marker: PhantomData,
        }
    }
}

// entry point to builder for small alphabets
impl<'t, I: SmallAlphabet>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, Undecided>
{
    pub fn for_text(text: &'t [I]) -> Self {
        Self {
            text: Some(Either::Left(text)),
            ..Self::init()
        }
    }
}

// entry point to builder for large alphabets
impl<'t, I: LargeAlphabet>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, Undecided>
{
    pub fn for_text_mut(text: &'t mut [I]) -> Self {
        Self {
            text: Some(Either::Right(text)),
            ..Self::init()
        }
    }
}

// second choice: output type and buffer mode
impl<'t, I: InputElement>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, Undecided>
{
    pub fn in_borrowed_buffer<'s, O>(
        self,
        suffix_array_buffer: &'s mut [O],
    ) -> SuffixArrayConstruction<'static, 's, 't, I, O, BorrowedBuffer, Undecided>
    where
        O: IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            suffix_array_buffer: Some(suffix_array_buffer),
            text: self.text,
            ..SuffixArrayConstruction::init()
        }
    }

    pub fn in_owned_buffer<O>(
        self,
    ) -> SuffixArrayConstruction<'static, 'static, 't, I, O, OwnedBuffer, Undecided>
    where
        O: IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            text: self.text,
            ..SuffixArrayConstruction::init()
        }
    }

    pub fn in_owned_buffer32(
        self,
    ) -> SuffixArrayConstruction<'static, 'static, 't, I, i32, OwnedBuffer, Undecided>
    where
        i32: IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            text: self.text,
            ..SuffixArrayConstruction::init()
        }
    }

    pub fn in_owned_buffer64(
        self,
    ) -> SuffixArrayConstruction<'static, 'static, 't, I, i64, OwnedBuffer, Undecided>
    where
        i64: IsValidOutputFor<I>,
    {
        SuffixArrayConstruction {
            text: self.text,
            ..SuffixArrayConstruction::init()
        }
    }
}

// third choice: threading
impl<'r, 's, 't, I: InputElement, O: OutputElement, B: BufferMode>
    SuffixArrayConstruction<'r, 's, 't, I, O, B, Undecided>
{
    pub fn single_threaded(self) -> SuffixArrayConstruction<'r, 's, 't, I, O, B, SingleThreaded> {
        SuffixArrayConstruction {
            text: self.text,
            suffix_array_buffer: self.suffix_array_buffer,
            ..SuffixArrayConstruction::init()
        }
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        self,
        thread_count: ThreadCount,
    ) -> SuffixArrayConstruction<'r, 's, 't, I, O, B, MultiThreaded> {
        SuffixArrayConstruction {
            text: self.text,
            suffix_array_buffer: self.suffix_array_buffer,
            thread_count,
            ..SuffixArrayConstruction::init()
        }
    }
}

impl<'r, 's, 't, I: SmallAlphabet, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'r, 's, 't, I, i32, B, P>
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
    pub unsafe fn with_alphabet_size(self, alphabet_size: AlphabetSize<O>) -> Self {
        Self {
            alphabet_size: alphabet_size.0,
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
    pub fn run(mut self) -> Result<SuffixArrayWithText<'s, 't, I, O, B>, LibsaisError> {
        let text_len = self.text().len();
        let mut suffix_array =
            OwnedOrBorrowed::take_buffer_or_allocate(self.suffix_array_buffer.take(), || {
                allocate_suffix_array_buffer::<I, O>(self.extra_space, text_len)
            });

        sais_safety_checks(
            self.text(),
            &suffix_array.buffer,
            &self.context,
            self.thread_count,
            self.generalized_suffix_array,
        );

        let (extra_space, text_len_output_type, num_threads, frequency_table_ptr) =
            cast_and_unpack_parameters(
                self.text().len(),
                &suffix_array.buffer,
                self.thread_count,
                self.frequency_table.take(),
            );

        // SAFETY:
        // buffer lens are safety checked (text and suffix array) with extra space in mind
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size (only small alphabets).
        // alphabet size was either set as the max of the text + 1 or claimed to be
        // correct in an unsafe function (only large alphabets).
        // context must be of the correct type, because the API is typesafe and the parallelism decision was
        // forced to happen before the context was supplied.
        let res = match self.text.as_mut() {
            None => unreachable!("There always needs to be a text provided for this object."),
            Some(Either::Left(text)) => {
                // small alphabets
                unsafe {
                    SmallAlphabetFunctionsDispatch::<I, O, P>::libsais(
                        text.as_ptr(),
                        suffix_array.buffer.as_mut_ptr(),
                        text_len_output_type,
                        extra_space,
                        frequency_table_ptr,
                        num_threads,
                        self.generalized_suffix_array,
                        self.context.take().map(|ctx| ctx.as_mut_ptr()),
                    )
                }
            }
            Some(Either::Right(text)) => {
                let alphabet_size: O = match self.alphabet_size {
                    AlphabetSizeInner::ComputeFromMaxOfText => {
                        compute_and_validate_alphabet_size(text).unwrap_or_else(|e| panic!("{e}"))
                    }
                    AlphabetSizeInner::Fixed { value } => value,
                };

                unsafe {
                    LargeAlphabetFunctionsDispatch::<I, O, P>::libsais_large_alphabet(
                        (*text).as_mut_ptr(),
                        suffix_array.buffer.as_mut_ptr(),
                        text_len_output_type,
                        alphabet_size,
                        extra_space,
                        num_threads,
                    )
                }
            }
        }
        .into_empty_sais_result();

        suffix_array.shorten_buffer_to(text_len);

        res.map(|_| SuffixArrayWithText {
            suffix_array,
            text: self.take_text(),
            is_generalized_suffix_array: self.generalized_suffix_array,
        })
    }

    fn text(&self) -> &[I] {
        self.text.as_ref().unwrap()
    }

    fn take_text(&mut self) -> &'t [I] {
        match self.text.take().unwrap() {
            Either::Left(t) => t,
            Either::Right(t) => t,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
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
    ) -> PlcpConstruction<'static, 's, 't, I, O, SaB, OwnedBuffer, Undecided> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtraSpace {
    None,
    Recommended,
    Fixed { value: usize },
}

impl ExtraSpace {
    fn compute_buffer_size<I: InputElement, O: OutputElement>(&self, text_len: usize) -> usize {
        match *self {
            ExtraSpace::None => text_len,
            ExtraSpace::Recommended => {
                if text_len <= 10_000 {
                    text_len
                } else {
                    let max_buffer_len_in_usize = O::MAX.into() as usize;
                    let desired_buffer_len = text_len + I::RECOMMENDED_EXTRA_SPACE;

                    if desired_buffer_len <= max_buffer_len_in_usize {
                        desired_buffer_len
                    } else if text_len <= max_buffer_len_in_usize {
                        max_buffer_len_in_usize
                    } else {
                        // if text_len was already too big, just return in and let safety checks later handle it
                        text_len
                    }
                }
            }
            ExtraSpace::Fixed { value } => text_len + value,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AlphabetSize<O: OutputElement>(AlphabetSizeInner<O>);

impl<O: OutputElement> AlphabetSize<O> {
    pub fn new(value: O) -> Self {
        Self(AlphabetSizeInner::Fixed { value })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum AlphabetSizeInner<O: OutputElementOrUndecided> {
    ComputeFromMaxOfText,
    Fixed { value: O },
}

pub(crate) fn allocate_suffix_array_buffer<I: InputElement, O: OutputElement>(
    extra_space_in_buffer: ExtraSpace,
    text_len: usize,
) -> Vec<O> {
    let buffer_len = extra_space_in_buffer.compute_buffer_size::<I, O>(text_len);
    vec![O::ZERO; buffer_len]
}

pub(crate) fn sais_safety_checks<I: InputElement, O: OutputElement, P: Parallelism>(
    text: &[I],
    suffix_array_buffer: &[O],
    context: &Option<&mut Context<I, O, P>>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
) {
    // the try_into implementations fail exactly when the value is too large for the respective libsais version
    let Ok(_): Result<O, _> = text.len().try_into() else {
        panic!(
            "The text is too long for the chosen output type. Text len: {}, Max allowed len: {}",
            text.len(),
            O::MAX
        );
    };

    let Ok(_): Result<O, _> = suffix_array_buffer.len().try_into() else {
        panic!(
            "The suffix array buffer is too long for chosen output type. Buffer len: {}, Max allowed len: {}",
            suffix_array_buffer.len(),
            O::MAX
        );
    };

    assert!(
        suffix_array_buffer.len() >= text.len(),
        "suffix_array_buffer must be at least as large as text"
    );

    if let Some(context) = context {
        assert_eq!(
            context.num_threads(),
            thread_count.value,
            "context needs to have the same number of threads as this config"
        );
    }

    if generalized_suffix_array && let Some(c) = text.last() {
        assert!(
            (*c).into() == 0i64,
            "For the generalized suffix array, the last character of the text needs to be 0 (not ASCII '0')"
        );
    }
}

pub(crate) fn cast_and_unpack_parameters<O: OutputElement>(
    text_len: usize,
    suffix_array_buffer: &[O],
    thread_count: ThreadCount,
    frequency_table: Option<&mut [O]>,
) -> (O, O, O, *mut O) {
    // all of these casts should succeed after the safety checks
    let extra_space = (suffix_array_buffer.len() - text_len).try_into().unwrap();
    let text_len = O::try_from(text_len).unwrap();
    let num_threads = O::try_from(thread_count.value as usize).unwrap();

    let frequency_table_ptr =
        frequency_table.map_or(std::ptr::null_mut(), |freq| freq.as_mut_ptr());

    (extra_space, text_len, num_threads, frequency_table_ptr)
}

// Computes the maximum value of the text and guarantees that all value are in the range
// [0, max_value]. Therefore the alphabet size returned is max_value + 1.
// Therefore the maximum value also has to be smaller than the maximum allowed value of `O`.
pub(crate) fn compute_and_validate_alphabet_size<I: InputElement, O: OutputElement>(
    text: &[I],
) -> Result<O, &'static str> {
    let mut min = I::ZERO;
    let mut max = I::ZERO;

    for c in text {
        min = min.min(*c);
        max = max.max(*c);
    }

    if min < I::ZERO {
        Err("Text cannot contain negative chars")
    } else {
        let found_max: i64 = max.into();
        let max_allowed: i64 = O::MAX.into();

        if found_max == max_allowed {
            Err("Text cannot contain the maximum value as a character")
        } else {
            Ok(O::try_from(found_max as usize + 1).unwrap())
        }
    }
}
