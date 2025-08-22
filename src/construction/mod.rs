pub mod bwt;
pub mod lcp;
pub mod plcp;
pub mod suffix_array;
pub mod unbwt;

use crate::{context::Context, type_model::*};

// -------------------- free helper functions for all configs --------------------
fn allocate_suffix_array_buffer<I: InputElement, O: OutputElement>(
    extra_space_in_buffer: ExtraSpace,
    text_len: usize,
) -> Vec<O> {
    let buffer_len = extra_space_in_buffer.compute_buffer_size::<I, O>(text_len);
    vec![O::ZERO; buffer_len]
}

pub(crate) fn free_extra_space<T>(suffix_array_buffer: &mut Vec<T>, text_len: usize) {
    suffix_array_buffer.truncate(text_len);
    suffix_array_buffer.shrink_to_fit();
}

fn cast_and_unpack_parameters<O: OutputElement>(
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

fn sais_safety_checks<I: InputElement, O: OutputElement, P: Parallelism>(
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
            c.clone().into() == 0i64,
            "For the generalized suffix array, the last character of the text needs to be 0 (not ASCII '0')"
        );
    }
}

fn aux_indices_safety_checks_and_cast_sampling_rate<O: OutputElement>(
    text_len: usize,
    aux_indices_buffer: &[O],
    aux_indices_sampling_rate: AuxIndicesSamplingRate,
) -> O {
    assert_eq!(
        aux_indices_buffer.len(),
        aux_indices_sampling_rate.aux_indices_buffer_size(text_len)
    );

    O::try_from(aux_indices_sampling_rate.value)
        .expect("Auxiliary indices sampling rate needs to fit into output element type")
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
    pub(crate) value: u16,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlphabetSize {
    ComputeFromMaxOfText,
    Fixed { value: usize },
}

trait IntoSaisResult {
    fn into_empty_sais_result(self) -> Result<(), SaisError>;

    fn into_primary_index_sais_result(self) -> Result<usize, SaisError>;
}

impl<O: OutputElement> IntoSaisResult for O {
    fn into_empty_sais_result(self) -> Result<(), SaisError> {
        let return_code: i64 = self.into();

        if return_code != 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(())
        }
    }

    fn into_primary_index_sais_result(self) -> Result<usize, SaisError> {
        let return_code: i64 = self.into();

        if return_code < 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(return_code as usize)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AuxIndicesSamplingRate {
    value: usize,
}

impl AuxIndicesSamplingRate {
    pub fn value(&self) -> usize {
        self.value
    }

    fn aux_indices_buffer_size(self, text_len: usize) -> usize {
        if text_len == 0 {
            0
        } else {
            (text_len - 1) / self.value + 1
        }
    }
}

impl From<usize> for AuxIndicesSamplingRate {
    fn from(value: usize) -> Self {
        if value < 2 {
            panic!("Aux sampling rate must be greater than 1");
        } else if value.count_ones() != 1 {
            panic!("Aux sampling rate must be a power of two");
        } else {
            Self { value }
        }
    }
}
