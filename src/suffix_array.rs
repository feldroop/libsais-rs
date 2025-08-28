/*!
 * Construct the (generalized) [suffix array] for a text using [`SuffixArrayConstruction`].
 *
 * `libsais` implements suffix array construction based on the [Suffix Array Induced Sort] (SAIS) algorithm.
 * It is a linear-time algorithm and only needs the suffix array itself as memory in most cases.
 *
 * The entry point to the API is the [`SuffixArrayConstruction`] builder-like struct. It is always required to
 * pass the input text, register the output element and make a decision about parallelisation. Further configuration
 * options include supplying an output buffer, [`context`], metadata about the input text, and instructing the library
 * to pass additional memory to the algorithm.
 *
 * The following is a fully-configured example of the suffix array construction for `u8`/`u16`-based alphabets:
 * ```
 * use libsais::{context::Context, SuffixArrayConstruction};
 *
 * let mut context = Context::new_single_threaded();
 *
 * let text = b"aaaaaaaaaa".as_slice();
 * let mut frequency_table = [0; 256];
 * frequency_table[b'a' as usize] = 10;
 *
 * // additional space in the buffer is automatically passed to the algorithm
 * let mut my_suffix_array_buffer = vec![0;15];
 *
 * let mut construction = SuffixArrayConstruction::for_text(&text)
 *     .in_borrowed_buffer(&mut my_suffix_array_buffer)
 *     .single_threaded()
 *     // .with_extra_space_in_buffer(ExtraSpace::Fixed { value: 5 }) <-- only for owned buffers
 *     .with_context(&mut context);
 *
 * // SAFETY: the frequency table for the example is correct
 * unsafe {
 *     construction = construction.with_frequency_table(&mut frequency_table);
 * }
 *
 * // after this, the suffix array will be stored in the beginning of my_suffix_array_buffer
 * let res = construction.run().unwrap();
 * println!("{:?}", res.suffix_array());
 * ```
 *
 * # Sentinel Convention and Suffix Array Length
 *
 * In the literature and in some applications, the input texts to suffix array construction algorithms are
 * assumed to be terminated by a unique, lexicographically smallest character. It is usually denoted by $ and
 * implemented by the zero byte.
 *
 * `libsais` does not have this requirement, but sorts suffixes as if such a
 * character were present at the end of the text. Therefore, the resulting suffix array has the same
 * length as the text. When using a borrowed output buffer, it has to be at least as long as the text.
 *
 * # Return Type and PLCP
 *
 * The read-only return type of [`SuffixArrayConstruction::run`], bundles the suffix array
 * and a reference to the input text. It is generic over whether an owned or borrowed suffix array buffer is used.
 * The object can be destructured into parts or used to safely compute a permuted longest common prefix (PLCP) array.
 *
 * An example of using the return type to obtain a PLCP and then an LCP array can be found
 * [here](https://github.com/feldroop/libsais-rs/blob/master/examples/the_lcp_ladder.rs).
 *
 * # Large Alphabets
 *
 * For `i32`/`i64`-based texts, a mutable reference to the input text is required. If the algorithm executes without
 * errors, the text will be returned to its initial state. Negative values in the input text are NOT allowed.
 *
 * Additionally, it is strongly recommended to pass an alphabet size using
 * [`SuffixArrayConstruction::with_alphabet_size`]. Otherwise, the library will inject a linear scan of the
 * text to determine a suitable alphabet size and guarantee that no negative values exist. The memory usage of
 * the algorithm is linear in the alphabet size.
 *
 * The largest value in the text plus one is a lower bound for the alphabet size. It is therefore wasteful
 * to use this algorithm on a text with a large maximum value, when many values smaller than the maximum do
 * not occur in the text. In such a scenario, mapping the text into the range [0, k) is a good option,
 * where k is the number of distinct values in the text.
 *
 * An example of using large alphabets with further explanations can be found
 * [here](https://github.com/feldroop/libsais-rs/blob/master/examples/large_alphabet_suffix_array.rs).
 *
 * # Generalized Suffix Array
 *
 * The generalized suffix array is a suffix array for a set of texts `{t1, t2, ..., tn}`. The set of all
 * suffixes of all texts is sorted to obtain this data structure. In theory, the array then contains tuples of
 * indices of the suffix and of the text the suffix belongs to.
 *
 * The generalized suffix array can be simulated in practice by concatenating the text using unique separators
 * and then constructing a normal suffix array for the concatenated text. The concatenation works like this:
 * `t = t1 $1 t2 $2 ... tn $n`, with `$1 < $2 < ... < $n`. `libsais` allows implementing this behavior by using the
 * zero byte (not ASCII '0') for every separator, like so: `t0 = t1 0 t2 0 ... tn 0`. In this API wrapper, simply
 * use the [`SuffixArrayConstruction::generalized_suffix_array`] function to activate this behavior.
 *
 * It would be possible to construct a normal suffix array of `t0` without this flag and obtain
 * a suffix array very similar to the generalized suffix array. However, the tie-breaking behavior
 * between equivalent suffixes of different texts would then be unpredicatble, because the zero bytes
 * would not be implicitly treated as unique, ordered separators.
 *
 * An example of creating a generalized suffix array of multiple texts can be found
 * [here](https://github.com/feldroop/libsais-rs/blob/master/examples/fully_configured_generalized_suffix_array.rs).
 *
 * [suffix array]: https://en.wikipedia.org/wiki/Suffix_array
 * [Suffix Array Induced Sort]: https://www.doi.org/10.1109/TC.2010.188
 * [`context`]: super::context
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
    typestate::{
        BorrowedBuffer, BufferMode, BufferModeOrUndecided, OutputElementOrUndecided, OwnedBuffer,
        Parallelism, ParallelismOrUndecided, SingleThreaded, Undecided,
    },
};

#[cfg(feature = "openmp")]
use crate::typestate::MultiThreaded;

use std::marker::PhantomData;

/// One of the two main entry points of this library, for constructing suffix arrays.
///
/// See [`suffix_array`](self) for details.
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

impl<'t, I: SmallAlphabet>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, Undecided>
{
    /// The first method to call for `u8`/`u16`-based texts.
    ///
    /// The text has to be at most as long as the maximum value of the output element type you will choose.
    pub fn for_text(text: &'t [I]) -> Self {
        Self {
            text: Some(Either::Left(text)),
            ..Self::init()
        }
    }
}

impl<'t, I: LargeAlphabet>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, Undecided>
{
    /// The first method to call for `i32`/`i64`-based texts.
    ///
    /// The text has to be at most as long as the maximum value of the output element type you will choose.
    pub fn for_text_mut(text: &'t mut [I]) -> Self {
        Self {
            text: Some(Either::Right(text)),
            ..Self::init()
        }
    }
}

impl<'t, I: InputElement>
    SuffixArrayConstruction<'static, 'static, 't, I, Undecided, Undecided, Undecided>
{
    /// Provide a buffer to the library in which the suffix array will be stored.
    ///
    /// The buffer has to be at least as large as the text, but at most as large as the maximum value
    /// of the output element type. Additional space might be used by the algorithm for better performance.
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

    /// Inform the library of your desired output element type,
    /// if you want to obtain the suffix array in a [`Vec`].
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

    /// Inform the library that you want to obtain the suffix array in a [`Vec<i32>`].
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

    /// Inform the library that you want to obtain the suffix array in a [`Vec<i64>`].
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

impl<'r, 's, 't, I: SmallAlphabet, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'r, 's, 't, I, O, B, P>
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

    /// Construct the generalized suffix array, which is the suffix array of a set of strings.
    ///
    /// When using this mode, the last character of the texts must be 0 (not ASCII '0').
    ///
    /// See [`suffix_array`](self#generalized-suffix-array) for details.
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
    /// Supply the algorithm with an alphabet size for large alphabets.
    ///
    /// See [`suffix_array`](self#large-alphabets) for details.
    ///
    /// # Safety
    ///
    /// By calling this function you are asserting that all values of the text
    /// you are using are in the range [0, alphabet_size) and that there are no negative values.
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
    /// Provide the algorithm with additional memory.
    ///
    /// This might slightly improve the performance in some cases. When using an owned buffer,
    /// this is automatically determined. The default is [`ExtraSpace::Recommended`].
    ///
    /// The extra space may not make the suffix array buffer larger than the maximum allowed value when
    /// using [`ExtraSpace::Fixed`].
    pub fn with_extra_space_in_buffer(self, extra_space: ExtraSpace) -> Self {
        Self {
            extra_space,
            ..self
        }
    }
}

impl<'r, 's, 't, I: InputElement, O: OutputElement, B: BufferMode, P: Parallelism>
    SuffixArrayConstruction<'r, 's, 't, I, O, B, P>
{
    /// Construct the suffix array for the given text.
    ///
    /// # Panics
    ///
    /// If any of the requirements of the methods called before are not met.
    ///
    /// # Returns
    ///
    /// An error or a type that bundles the suffix array with a reference to the text.
    /// See [`suffix_array`](self#return-type-and-plcp) for details.
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

/// The read-only return type of a suffix array construction.
///
/// It keeps a reference to the text to allow safely constructing a PLCP array.
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
    /// Construct this type without going through a [`SuffixArrayConstruction`] or by using the parts
    /// obtained by [`Self::into_parts`].
    ///
    /// # Safety
    ///
    /// You are claiming that the suffix array is correct for the text according to the conventions of `libsais`
    /// and that the indicator for the generalized suffix array is correct.
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

/// The extra space to give the algorithm when using an owned buffer to (maybe) improve performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtraSpace {
    None,
    /// The recommended extra space is 0 for `u8`/`u16`-based texts and for texts of length smaller than
    /// or equal to 20K. For larger, `i32`/`i64`-based texts, it is 6K.
    ///
    /// When the extra space would make the buffer larger
    /// than the maximum of the output element type, the maximum allowed buffer size is chosen.
    Recommended,
    Fixed {
        value: usize,
    },
}

impl ExtraSpace {
    fn compute_buffer_size<I: InputElement, O: OutputElement>(&self, text_len: usize) -> usize {
        match *self {
            ExtraSpace::None => text_len,
            ExtraSpace::Recommended => {
                if text_len <= 20_000 {
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

/// An alphabet size that is recommended to use when handling `i32`/`i64`-based texts.
///
/// See [`suffix_array`](self#large-alphabets) for details.
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
