use super::{AuxIndicesSamplingRate, ExtraSpace, IntoSaisResult, SaisError, ThreadCount};
use crate::context::SaisContext;
use crate::data_structures::{AuxIndices, Bwt, BwtWithAuxIndices};
use crate::type_model::*;

use std::marker::PhantomData;

pub struct BwtConstruction<
    'a,
    I: InputElement,
    O: OutputElement,
    B: BufferMode,
    P: Parallelism,
    A: AuxIndicesMode,
> {
    text: Option<&'a [I]>,
    bwt_buffer: Option<&'a mut [I]>,
    temporary_suffix_array_buffer: Option<&'a mut [O]>,
    frequency_table: Option<&'a mut [O]>,
    extra_space_temporary_suffix_array_buffer: ExtraSpace,
    thread_count: ThreadCount,
    context: Option<&'a mut I::SingleThreadedContext>,
    aux_indices_sampling_rate: Option<AuxIndicesSamplingRate<O>>,
    aux_indices_buffer: Option<&'a mut [O]>,
    _parallelism_marker: PhantomData<P>,
    _buffer_mode_marker: PhantomData<B>,
    _aux_indices_mode_marker: PhantomData<A>,
}

impl<'a, I: InputElement, O: OutputElement, B: BufferMode, P: Parallelism, A: AuxIndicesMode>
    BwtConstruction<'a, I, O, B, P, A>
{
    fn init() -> Self {
        Self {
            text: None,
            bwt_buffer: None,
            temporary_suffix_array_buffer: None,
            frequency_table: None,
            thread_count: ThreadCount::fixed(1),
            extra_space_temporary_suffix_array_buffer: ExtraSpace::Recommended,
            context: None,
            aux_indices_sampling_rate: None,
            aux_indices_buffer: None,
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
            _aux_indices_mode_marker: PhantomData,
        }
    }
}

impl<'a, I: InputElement, O: OutputElement, B1: BufferMode, P1: Parallelism, A1: AuxIndicesMode>
    BwtConstruction<'a, I, O, B1, P1, A1>
{
    fn into_other_marker_type<B2: BufferMode, P2: Parallelism, A2: AuxIndicesMode>(
        self,
    ) -> BwtConstruction<'a, I, O, B2, P2, A2> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_suffix_array_buffer: self.temporary_suffix_array_buffer,
            frequency_table: self.frequency_table,
            thread_count: self.thread_count,
            extra_space_temporary_suffix_array_buffer: self
                .extra_space_temporary_suffix_array_buffer,
            context: self.context,
            aux_indices_sampling_rate: self.aux_indices_sampling_rate,
            aux_indices_buffer: self.aux_indices_buffer,
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
            _aux_indices_mode_marker: PhantomData,
        }
    }
}

// entry point to builder
impl<'a, I: SmallAlphabet>
    BwtConstruction<'a, I, Undecided, Undecided, SingleThreaded, NoAuxIndices>
{
    pub fn for_text(text: &'a [I]) -> Self {
        Self {
            text: Some(text),
            ..Self::init()
        }
    }

    pub fn replace_text(
        text: &'a mut [I],
    ) -> BwtConstruction<'a, I, Undecided, BorrowedBuffer, SingleThreaded, NoAuxIndices> {
        BwtConstruction {
            bwt_buffer: Some(text),
            ..BwtConstruction::init()
        }
    }

    pub fn in_borrowed_buffer(
        self,
        bwt_buffer: &'a mut [I],
    ) -> BwtConstruction<'a, I, Undecided, BorrowedBuffer, SingleThreaded, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: Some(bwt_buffer),
            thread_count: self.thread_count,
            ..BwtConstruction::init()
        }
    }

    pub fn in_owned_buffer(
        self,
    ) -> BwtConstruction<'a, I, Undecided, OwnedBuffer, SingleThreaded, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            thread_count: self.thread_count,
            ..BwtConstruction::init()
        }
    }
}

// second choice: output type
impl<'a, I: SmallAlphabet, B: BufferMode>
    BwtConstruction<'a, I, Undecided, B, SingleThreaded, NoAuxIndices>
{
    pub fn with_borrowed_temporary_suffix_array_buffer<O: OutputElementDecided>(
        self,
        temporary_suffix_array_buffer: &'a mut [O],
    ) -> BwtConstruction<'a, I, O, B, SingleThreaded, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            temporary_suffix_array_buffer: Some(temporary_suffix_array_buffer),
            thread_count: self.thread_count,
            ..BwtConstruction::init()
        }
    }

    pub fn with_owned_temporary_suffix_array_buffer<O: OutputElementDecided>(
        self,
        extra_space: ExtraSpace,
    ) -> BwtConstruction<'a, I, O, B, SingleThreaded, NoAuxIndices> {
        BwtConstruction {
            text: self.text,
            bwt_buffer: self.bwt_buffer,
            thread_count: self.thread_count,
            extra_space_temporary_suffix_array_buffer: extra_space,
            ..BwtConstruction::init()
        }
    }
}

// optional choice at any time: with auxiliary indices
impl<'a, I: SmallAlphabet, O: OutputElementDecided, B: BufferMode, P: Parallelism>
    BwtConstruction<'a, I, O, B, P, NoAuxIndices>
{
    pub fn with_aux_indices(
        mut self,
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
    ) -> BwtConstruction<'a, I, O, B, P, AuxIndicesOwnedBuffer> {
        self.aux_indices_sampling_rate = Some(aux_indices_sampling_rate);
        self.into_other_marker_type()
    }

    pub fn with_aux_indices_in_buffer(
        mut self,
        aux_indices_sampling_rate: AuxIndicesSamplingRate<O>,
        aux_indices_buffer: &'a mut [O],
    ) -> BwtConstruction<'a, I, O, B, P, AuxIndicesBorrowedBuffer> {
        self.aux_indices_sampling_rate = Some(aux_indices_sampling_rate);
        self.aux_indices_buffer = Some(aux_indices_buffer);
        self.into_other_marker_type()
    }
}

// optional choice at any time: threading
impl<'a, I: SmallAlphabet, O: OutputElementDecided, B: BufferMode, A: AuxIndicesMode>
    BwtConstruction<'a, I, O, B, SingleThreaded, A>
{
    pub fn multi_threaded(
        mut self,
        thread_count: ThreadCount,
    ) -> BwtConstruction<'a, I, O, B, MultiThreaded, A> {
        self.thread_count = thread_count;
        self.into_other_marker_type()
    }
}

impl<'a, I: SmallAlphabet, B: BufferMode, A: AuxIndicesMode>
    BwtConstruction<'a, I, i32, B, SingleThreaded, A>
{
    /// Uses a context object that allows reusing memory across runs of the algorithm.
    /// Currently, this is only available for the single threaded 32-bit output version.
    pub fn with_context(self, context: &'a mut I::SingleThreadedContext) -> Self {
        Self {
            context: Some(context),
            ..self
        }
    }
}

impl<
    'a,
    I: SmallAlphabet,
    O: OutputElementDecided,
    B: BufferMode,
    P: Parallelism,
    A: AuxIndicesMode,
> BwtConstruction<'a, I, O, B, P, A>
{
    /// By calling this function you are claiming that the frequency table is valid for the text
    /// for which this config is used later. Otherwise there is not guarantee for correct behavior
    /// of the C library.
    pub unsafe fn with_frequency_table(self, frequency_table: &'a mut [O]) -> Self {
        assert_eq!(frequency_table.len(), I::FREQUENCY_TABLE_SIZE);

        Self {
            frequency_table: Some(frequency_table),
            ..self
        }
    }
}

impl<'a, I: SmallAlphabet, O: OutputElementDecided, P: Parallelism>
    BwtConstruction<'a, I, O, OwnedBuffer, P, NoAuxIndices>
{
    pub fn construct(self) -> Result<Bwt<I>, SaisError> {
        let mut bwt_buffer = super::allocate_bwt_buffer(self.text.unwrap().len());

        let mut construction = self.into_other_marker_type::<BorrowedBuffer, P, NoAuxIndices>();
        construction.bwt_buffer = Some(&mut bwt_buffer);

        construction
            .construct_in_borrowed_buffer()
            .map(|bwt_primary_index| Bwt {
                bwt_data: bwt_buffer,
                bwt_primary_index,
            })
    }
}

impl<'a, I: SmallAlphabet, O: OutputElementDecided, P: Parallelism>
    BwtConstruction<'a, I, O, BorrowedBuffer, P, NoAuxIndices>
{
    pub fn construct_in_borrowed_buffer(mut self) -> Result<Option<usize>, SaisError> {
        if let Some(text) = self.text
            && text.is_empty()
        {
            return Ok(None);
        }

        let bwt_buffer = self.bwt_buffer.take().unwrap();

        if self.text.is_none() && bwt_buffer.is_empty() {
            return Ok(None);
        }

        let text_len = self
            .text
            .map_or_else(|| bwt_buffer.len(), |text| text.len());

        let mut owned_temporary_suffix_array_buffer = Vec::new();
        let temporary_suffix_array_buffer = self.get_temporary_suffix_array_buffer_or_allocate(
            &mut owned_temporary_suffix_array_buffer,
            text_len,
        );

        if let Some(text) = self.text {
            super::safety_checks(
                text,
                temporary_suffix_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
            super::bwt_safety_checks(text, bwt_buffer);
        } else {
            super::safety_checks(
                bwt_buffer,
                temporary_suffix_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
        }

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            super::cast_and_unpack_parameters(
                text_len,
                temporary_suffix_array_buffer,
                self.thread_count,
                self.frequency_table.take(),
            );

        // this breaks Rusts aliasing rules, because bwt buffer migh have a mut ptr and a const ptr to it at the same time
        // However, these pointer are only used by a C function that explicitly allows this
        let text_ptr = self
            .text
            .map_or_else(|| bwt_buffer.as_ptr(), |text| text.as_ptr());

        // SAFETY:
        // text len is asserted to be in required range in safety checks.
        // bwt len is checked in bwt safety checks.
        // suffix array buffer is at least as large as text, asserted in safety checks.
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        // and the frequency table was asserted to be the correct size.
        // if there is a context it has the correct type, because that was claimed in an unsafe impl
        // for InputElementDecided.
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
}

impl<'a, I: SmallAlphabet, O: OutputElementDecided, P: Parallelism>
    BwtConstruction<'a, I, O, OwnedBuffer, P, AuxIndicesOwnedBuffer>
{
    pub fn construct_with_aux_indices(self) -> Result<BwtWithAuxIndices<I, O>, SaisError> {
        let aux_indices_sampling_rate = self.aux_indices_sampling_rate.unwrap();

        let (mut bwt_buffer, mut aux_indices_buffer) = super::allocate_bwt_with_aux_buffers::<I, O>(
            self.text.unwrap().len(),
            aux_indices_sampling_rate,
        );

        let mut construction =
            self.into_other_marker_type::<BorrowedBuffer, P, AuxIndicesBorrowedBuffer>();
        construction.bwt_buffer = Some(&mut bwt_buffer);
        construction.aux_indices_buffer = Some(&mut aux_indices_buffer);

        construction
            .construct_with_aux_indices_in_borrowed_and_borrowed_buffers()
            .map(|_| BwtWithAuxIndices {
                bwt_data: bwt_buffer,
                aux_indices: AuxIndices {
                    data: aux_indices_buffer,
                    sampling_rate: aux_indices_sampling_rate,
                },
            })
    }
}

impl<'a, I: SmallAlphabet, O: OutputElementDecided, P: Parallelism>
    BwtConstruction<'a, I, O, OwnedBuffer, P, AuxIndicesBorrowedBuffer>
{
    pub fn construct_with_aux_indices_in_owned_and_borrowed_buffers(
        self,
    ) -> Result<Bwt<I>, SaisError> {
        let mut bwt_buffer = super::allocate_bwt_buffer(self.text.unwrap().len());

        let mut construction =
            self.into_other_marker_type::<BorrowedBuffer, P, AuxIndicesBorrowedBuffer>();
        construction.bwt_buffer = Some(&mut bwt_buffer);

        construction
            .construct_with_aux_indices_in_borrowed_and_borrowed_buffers()
            .map(|_| Bwt {
                bwt_data: bwt_buffer,
                bwt_primary_index: None,
            })
    }
}

impl<'a, I: SmallAlphabet, O: OutputElementDecided, P: Parallelism>
    BwtConstruction<'a, I, O, BorrowedBuffer, P, AuxIndicesOwnedBuffer>
{
    pub fn construct_with_aux_indices_in_borrowed_and_owned_buffers(
        self,
    ) -> Result<Vec<O>, SaisError> {
        let text_len = self.text.map_or_else(
            || self.bwt_buffer.as_ref().unwrap().len(),
            |text| text.len(),
        );

        let mut aux_indices_buffer =
            super::allocate_aux_indices_buffer(text_len, self.aux_indices_sampling_rate.unwrap());

        let mut construction =
            self.into_other_marker_type::<BorrowedBuffer, P, AuxIndicesBorrowedBuffer>();
        construction.aux_indices_buffer = Some(&mut aux_indices_buffer);

        construction
            .construct_with_aux_indices_in_borrowed_and_borrowed_buffers()
            .map(|_| aux_indices_buffer)
    }
}

impl<'a, I: SmallAlphabet, O: OutputElementDecided, P: Parallelism>
    BwtConstruction<'a, I, O, BorrowedBuffer, P, AuxIndicesBorrowedBuffer>
{
    pub fn construct_with_aux_indices_in_borrowed_and_borrowed_buffers(
        mut self,
    ) -> Result<(), SaisError> {
        if let Some(text) = self.text
            && text.is_empty()
        {
            return Ok(());
        }

        let bwt_buffer = self.bwt_buffer.take().unwrap();

        if self.text.is_none() && bwt_buffer.is_empty() {
            return Ok(());
        }

        let text_len = self
            .text
            .map_or_else(|| bwt_buffer.len(), |text| text.len());

        let mut owned_temporary_suffix_array_buffer = Vec::new();
        let temporary_suffix_array_buffer = self.get_temporary_suffix_array_buffer_or_allocate(
            &mut owned_temporary_suffix_array_buffer,
            text_len,
        );

        if let Some(text) = self.text {
            super::safety_checks(
                text,
                temporary_suffix_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
            super::bwt_safety_checks(text, bwt_buffer);
        } else {
            super::safety_checks(
                bwt_buffer,
                temporary_suffix_array_buffer,
                &self.context,
                self.thread_count,
                false,
            );
        }

        let aux_indices_buffer = self.aux_indices_buffer.take().unwrap();
        let aux_indices_sampling_rate = self.aux_indices_sampling_rate.unwrap();

        super::aux_indices_safety_checks(text_len, aux_indices_buffer, aux_indices_sampling_rate);

        let (extra_space, text_len, num_threads, frequency_table_ptr) =
            super::cast_and_unpack_parameters(
                text_len,
                temporary_suffix_array_buffer,
                self.thread_count,
                self.frequency_table.take(),
            );

        // this breaks Rusts aliasing rules, because bwt buffer migh have a mut ptr and a const ptr to it at the same time
        // However, these pointer are only used by a C function that explicitly allows this
        let text_ptr = self
            .text
            .map_or_else(|| bwt_buffer.as_ptr(), |text| text.as_ptr());

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
}

impl<
    'a,
    I: SmallAlphabet,
    O: OutputElementDecided,
    B: BufferMode,
    P: Parallelism,
    A: AuxIndicesMode,
> BwtConstruction<'a, I, O, B, P, A>
{
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
