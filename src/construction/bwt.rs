use super::{
    AuxIndicesSamplingRate, ConstructionInit, ExtraSpace, IntoSaisResult, SaisError, ThreadCount,
};
use crate::context::SaisContext;
use crate::data_structures::{AuxIndices, Bwt, BwtWithAuxIndices};
use crate::type_model::*;

pub struct BwtConstruction<'a, P: Parallelism, I: InputElement, O: OutputElement> {
    frequency_table: Option<&'a mut [O]>,
    temporary_suffix_array_buffer: Option<&'a mut [O]>,
    extra_space_temporary_suffix_array_buffer: ExtraSpace,
    thread_count: ThreadCount,
    generalized_suffix_array: bool,
    context: Option<&'a mut I::SingleThreadedContext>,
    _parallelism_marker: std::marker::PhantomData<P>,
}

impl<'a, P: Parallelism, I: InputElement, O: OutputElement> ConstructionInit
    for BwtConstruction<'a, P, I, O>
{
    fn init() -> Self {
        Self {
            frequency_table: None,
            thread_count: P::DEFAULT_THREAD_COUNT,
            temporary_suffix_array_buffer: None,
            extra_space_temporary_suffix_array_buffer: ExtraSpace::Recommended,
            generalized_suffix_array: false,
            context: None,
            _parallelism_marker: std::marker::PhantomData,
        }
    }
}

super::construction_impl!(BwtConstruction);

impl<'a, P: Parallelism, I: SmallAlphabet, O: OutputElementDecided> BwtConstruction<'a, P, I, O> {
    pub fn with_temporary_suffix_array_buffer(
        self,
        temporary_suffix_array_buffer: &'a mut [O],
    ) -> Self {
        Self {
            temporary_suffix_array_buffer: Some(temporary_suffix_array_buffer),
            ..self
        }
    }

    /// If a temporary suffix array buffer is supplied, this value is ignored and instead inferred from
    /// the buffer
    pub fn with_temporary_suffix_array_buffer_extra_space(self, extra_space: ExtraSpace) -> Self {
        Self {
            extra_space_temporary_suffix_array_buffer: extra_space,
            ..self
        }
    }

    // -------------------- runners only bwt --------------------
    pub fn construct(self, text: &[I]) -> Result<Bwt<I>, SaisError> {
        let mut bwt_buffer = super::allocate_bwt_buffer(text.len());

        let res = self.construct_in_output_buffer(text, &mut bwt_buffer);

        res.map(|bwt_primary_index| Bwt {
            bwt_data: bwt_buffer,
            bwt_primary_index,
        })
    }

    pub fn construct_in_output_buffer(
        self,
        text: &[I],
        bwt_buffer: &mut [I],
    ) -> Result<Option<usize>, SaisError> {
        self.construct_in_output_buffer_text_opt(Some(text), bwt_buffer)
    }

    pub fn construct_in_text_buffer(self, text: &mut [I]) -> Result<Option<usize>, SaisError> {
        self.construct_in_output_buffer_text_opt(None, text)
    }

    fn construct_in_output_buffer_text_opt(
        mut self,
        text_opt: Option<&[I]>,
        bwt_buffer: &mut [I],
    ) -> Result<Option<usize>, SaisError> {
        if let Some(text) = text_opt
            && text.is_empty()
        {
            return Ok(None);
        }

        if text_opt.is_none() && bwt_buffer.is_empty() {
            return Ok(None);
        }

        let text_len = text_opt.map_or_else(|| bwt_buffer.len(), |text| text.len());

        let mut owned_temporary_suffix_array_buffer = Vec::new();
        let temporary_suffix_array_buffer = self.get_temporary_suffix_array_buffer_or_allocate(
            &mut owned_temporary_suffix_array_buffer,
            text_len,
        );

        if let Some(text) = text_opt {
            self.safety_checks(text, temporary_suffix_array_buffer);
            super::bwt_safety_checks(text, bwt_buffer);
        } else {
            self.safety_checks(bwt_buffer, temporary_suffix_array_buffer);
        }

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            self.cast_and_unpack_parameters(text_len, temporary_suffix_array_buffer);

        // this breaks Rusts aliasing rules, because bwt buffer migh have a mut ptr and a const ptr to it at the same time
        // However, these pointer are only used by a C function that explicitly allows this
        let text_ptr = text_opt.map_or_else(|| bwt_buffer.as_ptr(), |text| text.as_ptr());

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
                    text_ptr,
                    bwt_buffer.as_mut_ptr(),
                    temporary_suffix_array_buffer.as_mut_ptr(),
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                    num_threads,
                    self.context.map(|ctx| ctx.as_mut_ptr()),
                )
            }.into_primary_index_sais_result()
    }

    // -------------------- runners with aux indices --------------------
    pub fn construct_with_aux_indices(
        self,
        text: &[I],
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> Result<BwtWithAuxIndices<I, O>, SaisError> {
        let (mut bwt_buffer, mut aux_indices_buffer) =
            super::allocate_bwt_with_aux_buffers::<I, O>(text.len(), aux_indices_sampling_rate);

        let res = self.construct_with_aux_indices_in_output_buffers(
            text,
            &mut bwt_buffer,
            &mut aux_indices_buffer,
            aux_indices_sampling_rate,
        );

        res.map(|_| BwtWithAuxIndices {
            bwt_data: bwt_buffer,
            aux_indices: AuxIndices {
                data: aux_indices_buffer,
                sampling_rate: aux_indices_sampling_rate,
            },
        })
    }

    pub fn construct_with_aux_indices_in_text_buffer(
        self,
        text: &mut [I],
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> Result<AuxIndices<O>, SaisError> {
        let mut aux_indices_buffer =
            super::allocate_aux_indices_buffer::<O>(text.len(), aux_indices_sampling_rate);

        let res = self.construct_with_aux_indices_in_output_buffers_text_opt(
            None,
            text,
            &mut aux_indices_buffer,
            aux_indices_sampling_rate,
        );

        res.map(|_| AuxIndices {
            data: aux_indices_buffer,
            sampling_rate: aux_indices_sampling_rate,
        })
    }

    pub fn construct_with_aux_indices_in_text_and_output_buffers(
        self,
        text: &mut [I],
        aux_indices_buffer: &mut [O],
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> Result<(), SaisError> {
        self.construct_with_aux_indices_in_output_buffers_text_opt(
            None,
            text,
            aux_indices_buffer,
            aux_indices_sampling_rate,
        )
    }

    pub fn construct_with_aux_indices_in_output_buffers(
        self,
        text: &[I],
        bwt_buffer: &mut [I],
        aux_indices_buffer: &mut [O],
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> Result<(), SaisError> {
        self.construct_with_aux_indices_in_output_buffers_text_opt(
            Some(text),
            bwt_buffer,
            aux_indices_buffer,
            aux_indices_sampling_rate,
        )
    }

    fn construct_with_aux_indices_in_output_buffers_text_opt(
        mut self,
        text_opt: Option<&[I]>,
        bwt_buffer: &mut [I],
        aux_indices_buffer: &mut [O],
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> Result<(), SaisError> {
        if let Some(text) = text_opt
            && text.is_empty()
        {
            return Ok(());
        }

        if text_opt.is_none() && bwt_buffer.is_empty() {
            return Ok(());
        }

        let text_len = text_opt.map_or_else(|| bwt_buffer.len(), |text| text.len());

        let mut owned_temporary_suffix_array_buffer = Vec::new();
        let temporary_suffix_array_buffer = self.get_temporary_suffix_array_buffer_or_allocate(
            &mut owned_temporary_suffix_array_buffer,
            text_len,
        );

        if let Some(text) = text_opt {
            self.safety_checks(text, temporary_suffix_array_buffer);
            super::bwt_safety_checks(text, bwt_buffer);
        } else {
            self.safety_checks(bwt_buffer, temporary_suffix_array_buffer);
        }

        super::aux_indices_safety_checks(text_len, aux_indices_buffer, aux_indices_sampling_rate);

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            self.cast_and_unpack_parameters(text_len, temporary_suffix_array_buffer);

        // this breaks Rusts aliasing rules, because bwt buffer migh have a mut ptr and a const ptr to it at the same time
        // However, these pointer are only used by a C function that explicitly allows this
        let text_ptr = text_opt.map_or_else(|| bwt_buffer.as_ptr(), |text| text.as_ptr());

        // SAFETY:
        // text len is asserted to be in required range in safety checks
        // bwt len is checked in bwt safety checks
        // suffix array buffer is at least as large as text, asserted in safety checks
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size
        // if there is a context it has the correct type, because that was claimed in an unsafe impl
        // for InputElementDecided
        // aux indices also is the correct length as asserted by the safety checks
        unsafe {
                <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::SmallAlphabetFunctions::run_libsais_bwt_aux(
                    text_ptr,
                    bwt_buffer.as_mut_ptr(),
                    temporary_suffix_array_buffer.as_mut_ptr(),
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

    fn get_temporary_suffix_array_buffer_or_allocate<'b>(
        &mut self,
        owned_temporary_suffix_array_buffer: &'b mut Vec<O>,
        text_len: usize,
    ) -> &'b mut [O]
    where
        'a: 'b,
    {
        self.temporary_suffix_array_buffer
            .take()
            .unwrap_or_else(|| {
                *owned_temporary_suffix_array_buffer = super::allocate_suffix_array_buffer::<I, O>(
                    self.extra_space_temporary_suffix_array_buffer,
                    text_len,
                );

                owned_temporary_suffix_array_buffer
            })
    }
}
