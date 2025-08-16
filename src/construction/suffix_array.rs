use super::{AlphabetSize, ExtraSpace, IntoSaisResult, SaisError, ThreadCount};
use crate::context::SaisContext;
use crate::data_structures::SuffixArray;
use crate::helpers;
use crate::type_model::*;

use std::marker::PhantomData;

pub struct SuffixArrayConstruction<'a, P: Parallelism, I: InputElement, O: OutputElement> {
    frequency_table: Option<&'a mut [O]>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    alphabet_size: AlphabetSize,
    context: Option<&'a mut I::SingleThreadedContext>,
    _parallelism_marker: PhantomData<P>,
}

super::construction_impl!(
    SuffixArrayConstruction,
    // -------------------- first transition: choose input type (with large alphabets allowed) --------------------
    // impl block goes here to make it appear at the correct location in the docs
    impl<'a, P: Parallelism> SuffixArrayConstruction<'a, P, Undecided, Undecided> {
        pub fn input_and_output_32_bits(self) -> SuffixArrayConstruction<'a, P, i32, i32> {
            SuffixArrayConstruction {
                frequency_table: None,
                thread_count: self.thread_count,
                generalized_suffix_array: self.generalized_suffix_array,
                alphabet_size: self.alphabet_size,
                context: None,
                _parallelism_marker: PhantomData,
            }
        }

        pub fn input_and_output_64_bits(self) -> SuffixArrayConstruction<'a, P, i64, i64> {
            SuffixArrayConstruction {
                frequency_table: None,
                thread_count: self.thread_count,
                generalized_suffix_array: self.generalized_suffix_array,
                alphabet_size: self.alphabet_size,
                context: None,
                _parallelism_marker: PhantomData,
            }
        }
    }
);

// -------------------- operations and runners for only suffix array and small alphabets --------------------
impl<'a, P: Parallelism, I: SmallAlphabet, O: OutputElementDecided>
    SuffixArrayConstruction<'a, P, I, O>
{
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
    pub fn construct(
        self,
        text: &[I],
        extra_space_in_buffer: ExtraSpace,
    ) -> Result<SuffixArray<O>, SaisError> {
        let mut suffix_array_buffer =
            super::allocate_suffix_array_buffer::<I, O>(extra_space_in_buffer, text.len());

        let res = self.construct_in_output_buffer(text, &mut suffix_array_buffer);

        super::free_extra_space(&mut suffix_array_buffer, text.len());

        res.map(|_| SuffixArray {
            data: suffix_array_buffer,
        })
    }

    /// Construct the suffix array for the given text in the provided buffer.
    /// The buffer must be at least as large as the text. Additional space at the end
    /// will be used as extra space.
    /// The suffix array will be written to the start of the buffer
    pub fn construct_in_output_buffer(
        mut self,
        text: &[I],
        suffix_array_buffer: &mut [O],
    ) -> Result<(), SaisError> {
        if text.is_empty() {
            return Ok(());
        }

        self.safety_checks(text, suffix_array_buffer);

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            self.cast_and_unpack_parameters(text, suffix_array_buffer);

        // SAFETY:
        // text len is asserted to be in required range in safety checks
        // suffix array buffer is at least as large as text, asserted in safety checks
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size
        // if there is a context it has the correct type, because that was claimed in an unsafe impl
        // for InputElementDecided
        unsafe {
            <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::SmallAlphabetFunctions::run_libsais(
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
        .into_empty_sais_result()
    }
}

// -------------------- operations and runners for only suffix array and large alphabets --------------------
impl<'a, P: Parallelism, I: LargeAlphabet, O: OutputElementDecided>
    SuffixArrayConstruction<'a, P, I, O>
{
    /// By calling this function you are asserting that all values of the text
    /// you'll use for this configuration are in the range [0, alphabet_size)
    pub unsafe fn alphabet_size(self, alphabet_size: usize) -> Self {
        Self {
            alphabet_size: AlphabetSize::Fixed {
                value: alphabet_size,
            },
            ..self
        }
    }

    pub fn construct_with_large_alphabet(
        self,
        text: &mut [I],
        extra_space_in_buffer: ExtraSpace,
    ) -> Result<SuffixArray<O>, SaisError> {
        let mut suffix_array_buffer =
            super::allocate_suffix_array_buffer::<I, O>(extra_space_in_buffer, text.len());

        let res =
            self.construct_with_large_alphabet_in_output_buffer(text, &mut suffix_array_buffer);

        super::free_extra_space(&mut suffix_array_buffer, text.len());

        res.map(|_| SuffixArray {
            data: suffix_array_buffer,
        })
    }

    pub fn construct_with_large_alphabet_in_output_buffer(
        mut self,
        text: &mut [I],
        suffix_array_buffer: &mut [O],
    ) -> Result<(), SaisError> {
        if text.is_empty() {
            return Ok(());
        }

        self.safety_checks(text, suffix_array_buffer);

        let (extra_space, text_len, num_threads, _) =
            self.cast_and_unpack_parameters(text, suffix_array_buffer);

        // TODO compute stuff about alphabet size, with little safety check
        let alphabet_size: O = match self.alphabet_size {
            AlphabetSize::ComputeFromMaxOfText => {
                helpers::compute_and_validate_alphabet_size(text).unwrap_or_else(|e| panic!("{e}"))
            }
            AlphabetSize::Fixed { value } => value
                .try_into()
                .expect("The alphabet size needs to fit into the output type."),
        };

        // SAFETY:
        // text len is asserted to be in required range in safety checks
        // suffix array buffer is at least as large as text, asserted in safety checks
        // alphabet size was either set as the max of the text + 1 or claimed to be
        // correct in an unsafe function
        unsafe {
            <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::LargeAlphabetFunctions::run_libsais_large_alphabet(
                text.as_mut_ptr(),
                suffix_array_buffer.as_mut_ptr(),
                text_len,
                alphabet_size,
                extra_space,
                num_threads,
            )
        }.into_empty_sais_result()
    }
}
