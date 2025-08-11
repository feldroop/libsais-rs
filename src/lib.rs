mod context;
mod model;

use model::{MultiThreaded, Parallelism, SingleThreaded};

use std::{marker::PhantomData, ptr};

pub use context::{MultiThreadedSaisContext, SingleThreadedSaisContext};

pub use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

use crate::context::SaisContext;

/// The maximum text size that this library can handle when using i32-based buffers
pub const LIBSAIS_I32_MAXIMUM_TEXT_SIZE: usize = 2147483647;

// output structures: SA, SA+BWT, SA+BWT+AUX, GSA for multistring

// required extra config: aux -> sampling rate, alhpabet size for int array, unbwt primary index
// optional extra config: with context, unbwt context, omp, frequency table

// other queries: lcp from plcp and sa, plcp from sa/gsa and text, unbwt

// TODO num threads wrapper
// next steps: context, gsa, int input
// then: bwt + aux
// later: unbwt, plcp + lcp
pub struct SaisConfig<'a, P: Parallelism> {
    frequency_table: Option<&'a mut [i32; 256]>,
    num_threads: u16,
    context: Option<&'a mut P::Context>,
    _parallelism_marker: PhantomData<P>,
}

impl<'a> SaisConfig<'a, SingleThreaded> {
    pub fn single_threaded() -> Self {
        Self {
            frequency_table: None,
            num_threads: 1,
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
impl<'a> SaisConfig<'a, MultiThreaded> {
    pub fn multi_threaded() -> Self {
        Self {
            frequency_table: None,
            num_threads: 0, // TODO more expressive
            context: None,
            _parallelism_marker: PhantomData,
        }
    }

    /// Number of threads to use. Setting it to 0 will lead to the library choosing the
    /// number of threads (typically this will be equal to the available hardware parallelism).
    pub fn num_threads(self, num_threads: u16) -> Self {
        Self {
            num_threads,
            ..self
        }
    }
}

impl<'a, P: Parallelism> SaisConfig<'a, P> {
    /// By calling this function you are claiming that the frequency table is valid for the text
    /// for which this config is used later. Otherwise there is not guarantee for correct behavior
    /// of the C library.
    pub unsafe fn frequency_table(self, frequency_table: &'a mut [i32; 256]) -> Self {
        Self {
            frequency_table: Some(frequency_table),
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
        assert!(text.len() < LIBSAIS_I32_MAXIMUM_TEXT_SIZE);
        assert!(suffix_array_buffer.len() < LIBSAIS_I32_MAXIMUM_TEXT_SIZE);
        assert!(suffix_array_buffer.len() >= text.len());

        if let Some(context) = &self.context {
            assert_eq!(context.num_threads(), self.num_threads);
        }

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
                self.num_threads.into(),
                self.context.map(|ctx| ctx.as_mut_ptr()),
            )
        };

        if return_code != 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(())
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

    fn setup_example() -> (
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

    #[test]
    fn libsais_basic() {
        let (text, extra_space, mut frequency_table, mut ctx) = setup_example();

        let mut config = SaisConfig::single_threaded().with_context(&mut ctx);

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
    fn libsais_with_output_buffer() {
        let (text, extra_space, mut frequency_table, mut ctx) = setup_example();
        let buffer_size = text.len() + extra_space;
        let mut suffix_array_buffer = vec![0; buffer_size];

        let mut config = SaisConfig::single_threaded().with_context(&mut ctx);

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
        let (text, extra_space, mut frequency_table, _) = setup_example();

        let mut config = SaisConfig::multi_threaded().num_threads(4);

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
