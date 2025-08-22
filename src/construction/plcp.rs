use super::IntoSaisResult;
use crate::SaisError;
use crate::data_structures::{OwnedOrBorrowed, SuffixArrayWithPlcp};
#[cfg(feature = "openmp")]
use crate::type_model::MultiThreaded;
use crate::type_model::{
    BorrowedBuffer, BufferMode, InputElement, LcpFunctionsDispatch, LibsaisLcpFunctions,
    OutputElement, OwnedBuffer, Parallelism,
};
use crate::{ThreadCount, type_model::SingleThreaded};

use std::marker::PhantomData;

pub struct PlcpConstruction<
    'p,
    's,
    't,
    I: InputElement,
    O: OutputElement,
    PlcpB: BufferMode,
    SaB: BufferMode,
    P: Parallelism,
> {
    pub(crate) text: &'t [I],
    pub(crate) suffix_array_buffer: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) generalized_suffix_array: bool,
    pub(crate) plcp_buffer: Option<&'p mut [O]>,
    pub(crate) thread_count: ThreadCount,
    pub(crate) _parallelism_marker: PhantomData<P>,
    pub(crate) _plcp_buffer_mode_marker: PhantomData<PlcpB>,
}

#[cfg(feature = "openmp")]
impl<'p, 's, 't, I: InputElement, O: OutputElement, PlcpB: BufferMode, SaB: BufferMode>
    PlcpConstruction<'p, 's, 't, I, O, PlcpB, SaB, SingleThreaded>
{
    pub fn multithreaded(
        self,
        thread_count: ThreadCount,
    ) -> PlcpConstruction<'p, 's, 't, I, O, PlcpB, SaB, MultiThreaded> {
        PlcpConstruction {
            text: self.text,
            suffix_array_buffer: self.suffix_array_buffer,
            generalized_suffix_array: self.generalized_suffix_array,
            plcp_buffer: self.plcp_buffer,
            thread_count,
            _parallelism_marker: PhantomData,
            _plcp_buffer_mode_marker: PhantomData,
        }
    }
}

impl<'s, 't, I: InputElement, O: OutputElement, SaB: BufferMode, P: Parallelism>
    PlcpConstruction<'static, 's, 't, I, O, OwnedBuffer, SaB, P>
{
    pub fn in_borrowed_buffer<'p>(
        self,
        plcp_buffer: &'p mut [O],
    ) -> PlcpConstruction<'p, 's, 't, I, O, BorrowedBuffer, SaB, P> {
        PlcpConstruction {
            text: self.text,
            suffix_array_buffer: self.suffix_array_buffer,
            generalized_suffix_array: self.generalized_suffix_array,
            plcp_buffer: Some(plcp_buffer),
            thread_count: self.thread_count,
            _parallelism_marker: PhantomData,
            _plcp_buffer_mode_marker: PhantomData,
        }
    }
}

impl<
    'p,
    's,
    't,
    I: InputElement,
    O: OutputElement,
    PlcpB: BufferMode,
    SaB: BufferMode,
    P: Parallelism,
> PlcpConstruction<'p, 's, 't, I, O, PlcpB, SaB, P>
{
    pub fn run(mut self) -> Result<SuffixArrayWithPlcp<'p, 's, O, PlcpB, SaB>, SaisError> {
        let mut plcp = OwnedOrBorrowed::take_buffer_or_allocate(self.plcp_buffer.take(), || {
            vec![O::ZERO; self.text.len()]
        });

        self.construct_in_buffer(&mut plcp.buffer)
            .map(|_| SuffixArrayWithPlcp {
                plcp,
                suffix_array: self.suffix_array_buffer,
                is_generalized_suffix_array: self.generalized_suffix_array,
            })
    }

    fn construct_in_buffer(&mut self, plcp_buffer: &mut [O]) -> Result<(), SaisError> {
        assert_eq!(self.text.len(), plcp_buffer.len());
        assert_eq!(self.text.len(), self.suffix_array_buffer.buffer.len());

        // the try_into implementations fail exactly when the value is too large for the respective libsais version
        let Ok(text_len): Result<O, _> = self.text.len().try_into() else {
            panic!(
                "The text is too long for the chosen output type. Text len: {}, Max allowed len: {}",
                self.text.len(),
                O::MAX
            );
        };

        if self.generalized_suffix_array
            && let Some(c) = self.text.last()
        {
            assert!(
                c.clone().into() == 0i64,
                "For the generalized suffix array, the last character of the text needs to be 0 (not ASCII '0')"
            );
        }

        let num_threads = O::try_from(self.thread_count.value as usize).unwrap();

        // SAFETY: lens of buffers were checked
        // generalized suffix array expectations were checked
        // the content of the buffer must be correct, because this object could only be attained by
        // either claiming so in an unsafe fn or by constructing them using the appropriate functions of
        // this library
        unsafe {
            LcpFunctionsDispatch::<I, O, P>::libsais_plcp(
                self.text.as_ptr(),
                self.suffix_array_buffer.buffer.as_ptr(),
                plcp_buffer.as_mut_ptr(),
                text_len,
                num_threads,
                self.generalized_suffix_array,
            )
        }
        .into_empty_sais_result()
    }
}
