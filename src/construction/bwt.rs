use super::{AlphabetSize, AuxSamplingRate, ExtraSpace, IntoSaisResult, SaisError, ThreadCount};
use crate::context::SaisContext;
use crate::data_structures::{Bwt, BwtWithAuxIndices};
use crate::type_model::*;

pub struct BwtConstruction<'a, P: Parallelism, I: InputElement, O: OutputElement> {
    frequency_table: Option<&'a mut [O]>,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    alphabet_size: AlphabetSize,
    context: Option<&'a mut I::SingleThreadedContext>,
    _parallelism_marker: std::marker::PhantomData<P>,
}

super::construction_impl!(BwtConstruction,);

// -------------------- runners for suffix array and bwt --------------------
impl<'a, P: Parallelism, I: SmallAlphabet, O: OutputElementDecided> BwtConstruction<'a, P, I, O> {
    pub fn construct(
        self,
        text: &[I],
        extra_space_in_buffer: ExtraSpace,
    ) -> Result<Bwt<I>, SaisError> {
        let (mut suffix_array_buffer, mut bwt_buffer) =
            super::allocate_suffix_array_and_bwt_buffer::<I, O>(extra_space_in_buffer, text.len());

        let res = self.construct_in_output_buffers(text, &mut suffix_array_buffer, &mut bwt_buffer);

        res.map(|bwt_primary_index| Bwt {
            bwt_data: bwt_buffer,
            bwt_primary_index,
        })
    }

    pub fn construct_in_output_buffers(
        mut self,
        text: &[I],
        suffix_array_buffer: &mut [O],
        bwt_buffer: &mut [I],
    ) -> Result<Option<usize>, SaisError> {
        if text.is_empty() {
            return Ok(None);
        }

        self.safety_checks(text, suffix_array_buffer);
        super::bwt_safety_checks(text, bwt_buffer);

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            self.cast_and_unpack_parameters(text, suffix_array_buffer);

        // SAFETY:
        // text len is asserted to be in required range in safety checks
        // bwt len is checked in bwt safety checks
        // suffix array buffer is at least as large as text, asserted in safety checks
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size
        // if there is a context it has the correct type, because that was claimed in an unsafe impl
        // for InputElementDecided
        unsafe {
                <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::SmallAlphabetFunctions::run_libsais_bwt(
                    text.as_ptr(),
                    bwt_buffer.as_mut_ptr(),
                    suffix_array_buffer.as_mut_ptr(),
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                    num_threads,
                    self.context.map(|ctx| ctx.as_mut_ptr()),
                )
            }.into_primary_index_sais_result()
    }
}

// -------------------- runners for suffix array and bwt and aux indices --------------------
impl<'a, P: Parallelism, I: SmallAlphabet, O: OutputElementDecided> BwtConstruction<'a, P, I, O> {
    pub fn construct_with_aux_indices(
        self,
        text: &[I],
        extra_space_in_buffer: ExtraSpace,
        aux_indices_sampling_rate: AuxSamplingRate<O>,
    ) -> Result<BwtWithAuxIndices<I, O>, SaisError> {
        let (mut suffix_array_buffer, mut bwt_buffer, mut aux_indices_buffer) =
            super::allocate_suffix_array_and_bwt_and_aux_buffer::<I, O>(
                extra_space_in_buffer,
                text.len(),
                aux_indices_sampling_rate,
            );

        let res = self.construct_with_aux_indices_in_output_buffers(
            text,
            &mut suffix_array_buffer,
            &mut bwt_buffer,
            &mut aux_indices_buffer,
            aux_indices_sampling_rate,
        );

        res.map(|_| BwtWithAuxIndices {
            bwt_data: bwt_buffer,
            aux_indices_data: aux_indices_buffer,
        })
    }

    pub fn construct_with_aux_indices_in_output_buffers(
        mut self,
        text: &[I],
        suffix_array_buffer: &mut [O],
        bwt_buffer: &mut [I],
        aux_indices_buffer: &mut [O],
        aux_indices_sampling_rate: AuxSamplingRate<O>,
    ) -> Result<(), SaisError> {
        if text.is_empty() {
            return Ok(());
        }

        self.safety_checks(text, suffix_array_buffer);
        super::bwt_safety_checks(text, bwt_buffer);
        super::aux_indices_safety_checks(text.len(), aux_indices_buffer, aux_indices_sampling_rate);

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            self.cast_and_unpack_parameters(text, suffix_array_buffer);

        // SAFETY:
        // text len is asserted to be in required range in safety checks
        // bwt len is checked in bwt safety checks
        // suffix array buffer is at least as large as text, asserted in safety checks
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size
        // if there is a context it has the correct type, because that was claimed in an unsafe impl
        // for InputElementDecided
        unsafe {
                <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::SmallAlphabetFunctions::run_libsais_bwt_aux(
                    text.as_ptr(),
                    bwt_buffer.as_mut_ptr(),
                    suffix_array_buffer.as_mut_ptr(),
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                    aux_indices_sampling_rate.value,
                    aux_indices_buffer.as_mut_ptr(),
                    num_threads,
                    self.context.map(|ctx| ctx.as_mut_ptr()),
                )
            }.into_empty_sais_result()
    }
}
