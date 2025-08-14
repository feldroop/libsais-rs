mod helpers;
mod type_model;

use std::{marker::PhantomData, ptr};

use crate::{
    context::SaisContext,
    type_model::{InputDispatch, OutputDispatch, SmallInputBits},
};
use context::SingleThreadedSaisContext;
use type_model::{
    Input, InputBits, LibsaisFunctions, MultiThreaded, Output, OutputBits, Parallelism,
    SingleThreaded, Undecided,
};

pub mod context;
pub use helpers::concatenate_strings;

/// The version of the C library libsais wrapped by this crate
pub use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

/// The maximum text size that this library can handle when using i32-based buffers
pub const LIBSAIS_I32_OUTPUT_MAXIMUM_SIZE: usize = 2147483647;

/// The maximum text size that this library can handle when using i64-based buffers
pub const LIBSAIS_I64_OUTPUT_MAXIMUM_SIZE: usize = 9223372036854775807;

// functionality differences:
// libsais: full
// libsais16: (-2) no plcp_int_[omp] (int functions are repetitive anyway)
// libsais64, 16x64: (-2) np plcp_long_[omp], (-12) no*_ctx functions

// output structures: SA, SA+BWT, SA+BWT+AUX, GSA for multistring

// required extra config: aux -> sampling rate, alhpabet size for int array, unbwt primary index
// optional extra config: with context, unbwt context, omp, frequency table

// other queries: lcp from plcp and sa, plcp from sa/gsa and text, unbwt

// BIG TODOs:
//      bwt + aux
//      int input
//      unbwt, plcp + lcp

// SMALL TODOs:
//      redo context (generic and only callable when it is defined)
//      32/64 inputs with mutability issues,
//      num threads at correct position + refactor enum
//      recommended extra space
//      more tests
//      add try_API that is fallible on safety checks
//      seal traits
//      clean up using macros?

pub struct Sais<'a, P: Parallelism, I: Input, O: Output> {
    frequency_table: Option<&'a mut [O]>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    context: Option<&'a mut P::Context>,
    _parallelism_marker: PhantomData<P>,
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

// -------------------- entry point to builder single threaded --------------------
impl<'a> Sais<'a, SingleThreaded, Undecided, Undecided> {
    pub fn single_threaded() -> Self {
        Self {
            frequency_table: None,
            thread_count: ThreadCount::Fixed { value: 1 },
            generalized_suffix_array: false,
            context: None,
            _parallelism_marker: PhantomData,
            _input_marker: PhantomData,
            _output_marker: PhantomData,
        }
    }
    // TODO move context away because it only supports 8 and 16 bit inpus
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded version.
    pub fn with_context(self, context: &'a mut SingleThreadedSaisContext) -> Self {
        Self {
            context: Some(context),
            ..self
        }
    }
}

// -------------------- entry point to builder multithreaded --------------------
#[cfg(feature = "openmp")]
impl<'a> Sais<'a, MultiThreaded, Undecided, Undecided> {
    pub fn multi_threaded() -> Self {
        Self {
            frequency_table: None,
            thread_count: ThreadCount::OpenMpDefault,
            generalized_suffix_array: false,
            context: None,
            _parallelism_marker: PhantomData,
            _input_marker: PhantomData,
            _output_marker: PhantomData,
        }
    }

    // TODO move somewhere else where it can always be called
    /// Number of threads to use. Setting it to 0 will lead to the library choosing the
    /// number of threads (typically this will be equal to the available hardware parallelism).
    pub fn num_threads(self, thread_count: ThreadCount) -> Self {
        Self {
            thread_count,
            ..self
        }
    }
}

// -------------------- first transition: choose input type --------------------
impl<'a, P: Parallelism> Sais<'a, P, Undecided, Undecided> {
    pub fn input_8_bits(self) -> Sais<'a, P, u8, Undecided> {
        Sais {
            frequency_table: None,
            thread_count: self.thread_count,
            generalized_suffix_array: self.generalized_suffix_array,
            context: self.context,
            _parallelism_marker: PhantomData,
            _input_marker: PhantomData,
            _output_marker: PhantomData,
        }
    }

    pub fn input_16_bits(self) -> Sais<'a, P, u16, Undecided> {
        Sais {
            frequency_table: None,
            thread_count: self.thread_count,
            generalized_suffix_array: self.generalized_suffix_array,
            context: self.context,
            _parallelism_marker: PhantomData,
            _input_marker: PhantomData,
            _output_marker: PhantomData,
        }
    }

    // reintroduce once mutabiltiy issues are fixed
    // pub fn input_and_output_32_bits(self) -> Sais<'a, P, i32, i32> {
    //     Sais {
    //         frequency_table: None,
    //         thread_count: self.thread_count,
    //         generalized_suffix_array: self.generalized_suffix_array,
    //         context: self.context,
    //         _parallelism_marker: PhantomData,
    //         _input_marker: PhantomData,
    //         _output_marker: PhantomData,
    //     }
    // }

    // reintroduce once mutabiltiy issues are fixed
    // pub fn input_and_output_64_bits(self) -> Sais<'a, P, i64, i64> {
    //     Sais {
    //         frequency_table: None,
    //         thread_count: self.thread_count,
    //         generalized_suffix_array: self.generalized_suffix_array,
    //         context: self.context,
    //         _parallelism_marker: PhantomData,
    //         _input_marker: PhantomData,
    //         _output_marker: PhantomData,
    //     }
    // }
}

// -------------------- second transition: choose output type --------------------
impl<'a, P: Parallelism, I: InputBits> Sais<'a, P, I, Undecided> {
    pub fn output_32_bits(self) -> Sais<'a, P, I, i32> {
        Sais {
            frequency_table: None,
            thread_count: self.thread_count,
            generalized_suffix_array: self.generalized_suffix_array,
            context: self.context,
            _parallelism_marker: PhantomData,
            _input_marker: PhantomData,
            _output_marker: PhantomData,
        }
    }

    pub fn output_64_bits(self) -> Sais<'a, P, I, i64> {
        Sais {
            frequency_table: None,
            thread_count: self.thread_count,
            generalized_suffix_array: self.generalized_suffix_array,
            context: self.context,
            _parallelism_marker: PhantomData,
            _input_marker: PhantomData,
            _output_marker: PhantomData,
        }
    }
}

// -------------------- operations only defined for small input types --------------------
impl<'a, P: Parallelism, I: SmallInputBits, O: OutputBits> Sais<'a, P, I, O> {
    /// By calling this function you are claiming that the frequency table is valid for the text
    /// for which this config is used later. Otherwise there is not guarantee for correct behavior
    /// of the C library.
    pub unsafe fn frequency_table(self, frequency_table: &'a mut [O]) -> Self {
        assert_eq!(frequency_table.len(), I::FREQUENCY_TABLE_SIZE);

        Self {
            frequency_table: Some(frequency_table),
            ..self
        }
    }
}

impl<'a, P: Parallelism, I: InputBits, O: OutputBits> Sais<'a, P, I, O> {
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

    /// Construct the suffix array for the given text.
    pub fn run(self, text: &[I], extra_space_in_buffer: usize) -> Result<Vec<O>, SaisError> {
        let buffer_len = text.len() + extra_space_in_buffer;
        let mut suffix_array_buffer: Vec<O> = vec![O::try_from(0).unwrap(); buffer_len];

        let res: Result<(), SaisError> =
            self.run_with_output_buffer(text, &mut suffix_array_buffer);

        suffix_array_buffer.truncate(text.len());

        res.map(|_| suffix_array_buffer)
    }

    /// Construct the suffix array for the given text in the provided buffer.
    /// The buffer must be at least as large as the text. Additional space at the end
    /// will be used as extra space. The supplied extra space value will be ignored in this case.
    pub fn run_with_output_buffer(
        self,
        text: &[I],
        suffix_array_buffer: &mut [O],
    ) -> Result<(), SaisError> {
        self.safety_checks(text, suffix_array_buffer);

        let extra_space: O = (suffix_array_buffer.len() - text.len()).try_into().unwrap();

        let frequency_table_ptr = self
            .frequency_table
            .map_or(ptr::null_mut(), |freq| freq.as_mut_ptr());

        let text_len = O::try_from(text.len()).unwrap();
        let num_threads = O::try_from(self.thread_count.into_libsais_convention()).unwrap();

        // SAFETY:
        // text len is asserted to be in required range, which also makes the as i32 cast valid
        // suffix array buffer is asserted above to have the correct length
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size
        let return_code: i64 = unsafe {
            <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::Functions::run_libsais(
                text.as_ptr(),
                suffix_array_buffer.as_mut_ptr(),
                text_len,
                extra_space,
                frequency_table_ptr,
                num_threads,
                self.generalized_suffix_array,
                self.context.map(|ctx| ctx.as_mut_ptr()),
            )
        }
        .into();

        if return_code != 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(())
        }
    }

    fn safety_checks(&self, text: &[I], suffix_array_buffer: &mut [O]) {
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

        if let Some(context) = &self.context {
            assert_eq!(
                context.num_threads() as usize,
                self.thread_count.into_libsais_convention(),
                "context needs to have the same number of threads as this config"
            );
        }

        if self.generalized_suffix_array
            && let Some(c) = text.last()
        {
            assert!(
                c.clone().into() == 0i64,
                "For the generalized suffix array, the last character of the text needs to be 0 (not ASCII '0')"
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaisError {
    InvalidConfig,
    AlgorithmError,
    UnknownError,
}

impl SaisError {
    fn from_return_code(return_code: i64) -> Self {
        match return_code {
            0 => panic!("Return code does not indicate an error"),
            -1 => Self::InvalidConfig,
            -2 => Self::AlgorithmError,
            _ => Self::UnknownError,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadCount {
    OpenMpDefault,
    Fixed { value: u16 },
}

impl ThreadCount {
    pub fn fixed(thread_count: u16) -> Self {
        Self::Fixed {
            value: thread_count,
        }
    }

    fn into_libsais_convention(self) -> usize {
        match self {
            Self::OpenMpDefault => 0,
            Self::Fixed { value } => value.into(),
        }
    }
}
