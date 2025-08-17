pub mod bwt;
pub mod suffix_array;

use crate::type_model::*;

trait ConstructionInit {
    fn init() -> Self;
}

// I know this macro is a bit naughty and code sharing like this should preferably done with traits
// and default method implementations. However, with all of the typestate going on, that approach became
// involved quite quickly and made the docs look even more noisy than they are now.
macro_rules! construction_impl {
    ($struct_name:ident) => {
        // -------------------- entry point to builder single threaded --------------------
        impl<'a> $struct_name<'a, SingleThreaded, Undecided, Undecided> {
            pub fn single_threaded() -> Self {
                ConstructionInit::init()
            }
        }

        // -------------------- entry point to builder multithreaded --------------------
        #[cfg(feature = "openmp")]
        impl<'a> $struct_name<'a, MultiThreaded, Undecided, Undecided> {
            pub fn multi_threaded() -> Self {
                ConstructionInit::init()
            }
        }

        // -------------------- first transition: choose input type --------------------
        impl<'a, P: Parallelism> $struct_name<'a, P, Undecided, Undecided> {
            pub fn input_8_bits(
                self,
            ) -> $struct_name<'a, P, u8, Undecided> {
                ConstructionInit::init()
            }

            pub fn input_16_bits(
                self,
            ) -> $struct_name<'a, P, u16, Undecided> {
                ConstructionInit::init()
            }
        }

        // -------------------- second transition: choose output type --------------------
        impl<'a, P: Parallelism, I: InputElementDecided>
            $struct_name<'a, P, I, Undecided>
        {
            pub fn output_32_bits(self) -> $struct_name<'a, P, I, i32> {
                ConstructionInit::init()
            }

            pub fn output_64_bits(self) -> $struct_name<'a, P, I, i64> {
                ConstructionInit::init()
            }
        }

        // -------------------- Choose threads after choosing types and only with multithreaded config --------------------
        #[cfg(feature = "openmp")]
        impl<'a, I: InputElementDecided, O: OutputElementDecided>
            $struct_name<'a, MultiThreaded, I, O>
        {
            /// Number of threads to use. Setting it to 0 will lead to the library choosing the
            /// number of threads (typically this will be equal to the available hardware parallelism).
            pub fn num_threads(self, thread_count: ThreadCount) -> Self {
                Self {
                    thread_count,
                    ..self
                }
            }
        }

        // -------------------- support context only when it is implemented --------------------
        impl<'a, I: SmallAlphabet> $struct_name<'a, SingleThreaded, I, i32> {
            /// Uses a context object that allows reusing memory across runs of the algorithm.
            /// Currently, this is only available for the single threaded version.
            pub fn with_context(self, context: &'a mut I::SingleThreadedContext) -> Self {
                Self {
                    context: Some(context),
                    ..self
                }
            }
        }

        // -------------------- operations only defined for small input types and all result structures --------------------
        impl<'a, P: Parallelism, I: SmallAlphabet, O: OutputElementDecided>
            $struct_name<'a, P, I, O>
        {
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

        // -------------------- helper functions for all configs --------------------
        impl<'a, P: Parallelism, I: InputElementDecided, O: OutputElementDecided>
            $struct_name<'a, P, I, O>
        {
            fn cast_and_unpack_parameters(
                &mut self,
                text_len: usize,
                suffix_array_buffer: &[O],
            ) -> (O, O, O, *mut O) {
                // all of these casts should succeed after the safety checks
                let extra_space = (suffix_array_buffer.len() - text_len).try_into().unwrap();
                let text_len = O::try_from(text_len).unwrap();
                let num_threads = O::try_from(self.thread_count.value as usize).unwrap();

                let frequency_table_ptr = self
                    .frequency_table
                    .take()
                    .map_or(std::ptr::null_mut(), |freq| freq.as_mut_ptr());

                (extra_space, text_len, num_threads, frequency_table_ptr)
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
                        context.num_threads(),
                        self.thread_count.value,
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
    };
}

pub(crate) use construction_impl;

// -------------------- free helper functions for all configs --------------------
fn allocate_suffix_array_buffer<I: InputElementDecided, O: OutputElementDecided>(
    extra_space_in_buffer: ExtraSpace,
    text_len: usize,
) -> Vec<O> {
    let buffer_len = extra_space_in_buffer.compute_buffer_size::<I, O>(text_len);
    vec![O::try_from(0).unwrap(); buffer_len]
}

fn allocate_bwt_buffer<I: InputElementDecided>(text_len: usize) -> Vec<I> {
    vec![I::try_from(0).unwrap(); text_len]
}

fn allocate_bwt_with_aux_buffers<I: InputElementDecided, O: OutputElementDecided>(
    text_len: usize,
    aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
) -> (Vec<I>, Vec<O>) {
    (
        allocate_bwt_buffer(text_len),
        allocate_aux_indices_buffer(text_len, aux_indices_sampling_rate),
    )
}

fn allocate_aux_indices_buffer<O: OutputElementDecided>(
    text_len: usize,
    aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
) -> Vec<O> {
    vec![O::try_from(0).unwrap(); aux_indices_sampling_rate.aux_indices_buffer_size(text_len)]
}

fn free_extra_space<O: OutputElementDecided>(suffix_array_buffer: &mut Vec<O>, text_len: usize) {
    suffix_array_buffer.truncate(text_len);
    suffix_array_buffer.shrink_to_fit();
}

fn bwt_safety_checks<I: InputElementDecided>(text: &[I], bwt_buffer: &[I]) {
    assert_eq!(text.len(), bwt_buffer.len());
}

fn aux_indices_safety_checks<O: OutputElementDecided>(
    text_len: usize,
    aux_indices_buffer: &[O],
    aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
) {
    assert_eq!(
        aux_indices_buffer.len(),
        aux_indices_sampling_rate.aux_indices_buffer_size(text_len)
    );
}

// -------------------- various small structs and traits --------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaisError {
    InvalidConfig,
    AlgorithmError,
    UnknownError,
}

impl SaisError {
    fn from_return_code(return_code: i64) -> Self {
        println!("RETURN CODE: {return_code}");
        match return_code {
            0 => panic!("Return code does not indicate an error"),
            -1 => Self::InvalidConfig,
            -2 => Self::AlgorithmError,
            _ => Self::UnknownError,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ThreadCount {
    value: u16,
}

impl ThreadCount {
    pub const fn fixed(thread_count: u16) -> Self {
        if thread_count == 0 {
            panic!("Fixed thread count cannot be 0");
        }

        Self {
            value: thread_count,
        }
    }

    pub const fn openmp_default() -> Self {
        Self { value: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtraSpace {
    None,
    Recommended,
    Fixed { value: usize },
}

impl ExtraSpace {
    fn compute_buffer_size<I: InputElementDecided, O: OutputElementDecided>(
        &self,
        text_len: usize,
    ) -> usize {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlphabetSize {
    ComputeFromMaxOfText,
    Fixed { value: usize },
}

trait IntoSaisResult {
    fn into_empty_sais_result(self) -> Result<(), SaisError>;

    fn into_primary_index_sais_result(self) -> Result<Option<usize>, SaisError>;
}

impl<O: OutputElementDecided> IntoSaisResult for O {
    fn into_empty_sais_result(self) -> Result<(), SaisError> {
        let return_code: i64 = self.into();

        if return_code != 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(())
        }
    }

    fn into_primary_index_sais_result(self) -> Result<Option<usize>, SaisError> {
        let return_code: i64 = self.into();

        if return_code < 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(Some(return_code as usize))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AuxIndicesSamplingRate<O: OutputElementDecided> {
    value: O,
}

impl<O: OutputElementDecided> AuxIndicesSamplingRate<O> {
    fn aux_indices_buffer_size(self, text_len: usize) -> usize {
        let value_i64: i64 = self.value.into();
        let value_usize = value_i64 as usize;

        if text_len == 0 {
            0
        } else {
            (text_len - 1) / value_usize + 1
        }
    }
}

impl<O: OutputElementDecided> From<O> for AuxIndicesSamplingRate<O> {
    fn from(value: O) -> Self {
        let value_i64: i64 = value.into();

        if value_i64 < 2 {
            panic!("Aux sampling rate must be greater than 1");
        } else if value_i64.count_ones() != 1 {
            panic!("Aux sampling rate must be a power of two");
        } else {
            Self { value }
        }
    }
}
