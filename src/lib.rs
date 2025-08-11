mod helpers;
mod model;

use std::{marker::PhantomData, ptr};

use crate::context::SaisContext;
use context::SingleThreadedSaisContext;
use model::{MultiThreaded, Parallelism, SingleThreaded};

pub mod context;
pub use helpers::concatenate_strings_for_generalized_suffix_array;

/// The version of the C library libsais wrapped by this crate
pub use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

/// The maximum text size that this library can handle when using i32-based buffers
pub const LIBSAIS_I32_MAXIMUM_TEXT_SIZE: usize = 2147483647;

// output structures: SA, SA+BWT, SA+BWT+AUX, GSA for multistring

// required extra config: aux -> sampling rate, alhpabet size for int array, unbwt primary index
// optional extra config: with context, unbwt context, omp, frequency table

// other queries: lcp from plcp and sa, plcp from sa/gsa and text, unbwt

// next: bwt + aux
// later: unbwt, plcp + lcp
// finally: 16, 16x64, 64
pub struct Sais<'a, P: Parallelism> {
    frequency_table: Option<&'a mut [i32; 256]>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    context: Option<&'a mut P::Context>,
    _parallelism_marker: PhantomData<P>,
}

impl<'a> Sais<'a, SingleThreaded> {
    pub fn single_threaded() -> Self {
        Self {
            frequency_table: None,
            thread_count: ThreadCount::Fixed { value: 1 },
            generalized_suffix_array: false,
            context: None,
            _parallelism_marker: PhantomData,
        }
    }

    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded version.
    pub fn with_context(self, context: &'a mut SingleThreadedSaisContext) -> Self {
        Self {
            context: Some(context),
            ..self
        }
    }
}

#[cfg(feature = "openmp")]
impl<'a> Sais<'a, MultiThreaded> {
    pub fn multi_threaded() -> Self {
        Self {
            frequency_table: None,
            thread_count: ThreadCount::OpenMpDefault,
            generalized_suffix_array: false,
            context: None,
            _parallelism_marker: PhantomData,
        }
    }

    /// Number of threads to use. Setting it to 0 will lead to the library choosing the
    /// number of threads (typically this will be equal to the available hardware parallelism).
    pub fn num_threads(self, thread_count: ThreadCount) -> Self {
        Self {
            thread_count,
            ..self
        }
    }
}

impl<'a, P: Parallelism> Sais<'a, P> {
    /// By calling this function you are claiming that the frequency table is valid for the text
    /// for which this config is used later. Otherwise there is not guarantee for correct behavior
    /// of the C library.
    pub unsafe fn frequency_table(self, frequency_table: &'a mut [i32; 256]) -> Self {
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

    /// Construct the suffix array for the given text.
    pub fn run(self, text: &[u8], extra_space_in_buffer: usize) -> Result<Vec<i32>, SaisError> {
        let buffer_len = text.len() + extra_space_in_buffer;
        let mut suffix_array_buffer = vec![0; buffer_len];

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
        text: &[u8],
        suffix_array_buffer: &mut [i32],
    ) -> Result<(), SaisError> {
        self.safety_checks(text, suffix_array_buffer);

        let extra_space = (suffix_array_buffer.len() - text.len()) as i32;

        let frequency_table_ptr = self
            .frequency_table
            .map_or(ptr::null_mut(), |freq| freq.as_mut_ptr());

        // SAFETY:
        // text len is asserted to be in required range, which also makes the as i32 cast valid
        // suffix array buffer is asserted above to have the correct length
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        let return_code = unsafe {
            P::run_libsais(
                text.as_ptr(),
                suffix_array_buffer.as_mut_ptr(),
                text.len() as i32,
                extra_space,
                frequency_table_ptr,
                self.thread_count.into_libsais_convention(),
                self.generalized_suffix_array,
                self.context.map(|ctx| ctx.as_mut_ptr()),
            )
        };

        if return_code != 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(())
        }
    }

    fn safety_checks(&self, text: &[u8], suffix_array_buffer: &mut [i32]) {
        assert!(
            text.len() < LIBSAIS_I32_MAXIMUM_TEXT_SIZE,
            "text is too large for the basic version of libsais"
        );
        assert!(
            suffix_array_buffer.len() < LIBSAIS_I32_MAXIMUM_TEXT_SIZE,
            "suffix_array_buffer is too large for the basic version of libsais"
        );
        assert!(
            suffix_array_buffer.len() >= text.len(),
            "suffix_array_buffer must be at least as large as text"
        );

        if let Some(context) = &self.context {
            assert_eq!(
                context.num_threads() as i32,
                self.thread_count.into_libsais_convention(),
                "context needs to have the same number of threads as this config"
            );
        }

        if self.generalized_suffix_array
            && let Some(c) = text.last()
        {
            assert!(
                *c == 0,
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
    fn from_return_code(return_code: i32) -> Self {
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

    fn into_libsais_convention(self) -> i32 {
        match self {
            Self::OpenMpDefault => 0,
            Self::Fixed { value } => value.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_suffix_array(text: &[u8], maybe_suffix_array: &[i32]) -> bool {
        if text.is_empty() && maybe_suffix_array.is_empty() {
            return true;
        }

        for indices in maybe_suffix_array.windows(2) {
            let previous = indices[0] as usize;
            let current = indices[1] as usize;

            if &text[previous..] > &text[current..] {
                return false;
            }
        }

        true
    }

    fn is_generalized_suffix_array(concatenated_text: &[u8], maybe_suffix_array: &[i32]) -> bool {
        if concatenated_text.is_empty() && maybe_suffix_array.is_empty() {
            return true;
        }

        for indices in maybe_suffix_array.windows(2) {
            let previous = indices[0] as usize;
            let current = indices[1] as usize;

            // for the generalized suffix array, the zero char borders can be in a different order than
            // they would be in the normal suffix array
            if concatenated_text[previous] == 0 && concatenated_text[current] == 0 {
                continue;
            }

            if &concatenated_text[previous..] > &concatenated_text[current..] {
                return false;
            }
        }

        true
    }

    fn setup_basic_example() -> (
        &'static [u8; 11],
        usize,
        [i32; 256],
        SingleThreadedSaisContext,
    ) {
        let text = b"abababcabba";
        let extra_space = 10;
        let mut frequency_table = [0; 256];
        frequency_table[b'a' as usize] = 5;
        frequency_table[b'b' as usize] = 5;
        frequency_table[b'c' as usize] = 1;
        let ctx = SingleThreadedSaisContext::new();

        (text, extra_space, frequency_table, ctx)
    }

    fn setup_generalized_suffix_array_example()
    -> (Vec<u8>, usize, [i32; 256], SingleThreadedSaisContext) {
        let text = concatenate_strings_for_generalized_suffix_array([
            b"abababcabba".as_slice(),
            b"babaabccbac",
        ]);
        let extra_space = 20;
        let mut frequency_table = [0; 256];
        frequency_table[b'a' as usize] = 9;
        frequency_table[b'b' as usize] = 9;
        frequency_table[b'c' as usize] = 4;
        let ctx = SingleThreadedSaisContext::new();

        (text, extra_space, frequency_table, ctx)
    }

    #[test]
    fn libsais_basic() {
        let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

        let mut config = Sais::single_threaded().with_context(&mut ctx);

        // SAFETY: the frequency table defined above is valid
        unsafe {
            config = config.frequency_table(&mut frequency_table);
        }

        let suffix_array = config
            .run(text, extra_space)
            .expect("libsais should run without an error");

        assert!(is_suffix_array(text, &suffix_array));
    }

    #[test]
    fn libsais_generalized_suffix_array() {
        let (text, extra_space, mut frequency_table, mut ctx) =
            setup_generalized_suffix_array_example();

        let mut config = Sais::single_threaded()
            .generalized_suffix_array()
            .with_context(&mut ctx);

        // SAFETY: the frequency table defined above is valid
        unsafe {
            config = config.frequency_table(&mut frequency_table);
        }

        let suffix_array = config
            .run(&text, extra_space)
            .expect("libsais should run without an error");

        println!("{suffix_array:?}");

        assert!(is_generalized_suffix_array(&text, &suffix_array));
    }

    #[test]
    fn libsais_with_output_buffer() {
        let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
        let buffer_size = text.len() + extra_space;
        let mut suffix_array_buffer = vec![0; buffer_size];

        let mut config = Sais::single_threaded().with_context(&mut ctx);

        // SAFETY: the frequency table defined in the example is valid
        unsafe {
            config = config.frequency_table(&mut frequency_table);
        }

        let _ = config
            .run_with_output_buffer(text, &mut suffix_array_buffer)
            .expect("libsais should run without an error");

        assert!(is_suffix_array(text, &suffix_array_buffer[..text.len()]));
    }

    #[cfg(feature = "openmp")]
    #[test]
    fn libsais_omp() {
        let (text, extra_space, mut frequency_table, _) = setup_basic_example();

        let mut config = Sais::multi_threaded().num_threads(ThreadCount::OpenMpDefault);

        // SAFETY: the frequency table defined above is valid
        unsafe {
            config = config.frequency_table(&mut frequency_table);
        }

        let suffix_array = config
            .run(text, extra_space)
            .expect("libsais should run without an error");

        assert!(is_suffix_array(text, &suffix_array));
    }
}
