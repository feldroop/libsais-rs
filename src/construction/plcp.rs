use super::IntoSaisResult;
use crate::SaisError;
use crate::data_structures::{BorrowedPlcpWithBorrowedSuffixArray, PlcpWithBorrowedSuffixArray};
#[cfg(feature = "openmp")]
use crate::type_model::MultiThreaded;
use crate::type_model::{
    BorrowedBuffer, BufferMode, InputDispatch, InputElement, LcpFunctions, OutputDispatch,
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
    B: BufferMode,
    P: Parallelism,
> {
    pub(crate) text: &'t [I],
    pub(crate) suffix_array_buffer: &'s [O],
    pub(crate) generalized_suffix_array: bool,
    pub(crate) plcp_buffer: Option<&'p mut [O]>,
    pub(crate) thread_count: ThreadCount,
    pub(crate) _parallelism_marker: PhantomData<P>,
    pub(crate) _buffer_mode_marker: PhantomData<B>,
}

impl<'p, 's, 't, I: InputElement, O: OutputElement, B1: BufferMode, P1: Parallelism>
    PlcpConstruction<'p, 's, 't, I, O, B1, P1>
{
    fn into_other_marker_type<B2: BufferMode, P2: Parallelism>(
        self,
    ) -> PlcpConstruction<'p, 's, 't, I, O, B2, P2> {
        PlcpConstruction {
            text: self.text,
            suffix_array_buffer: self.suffix_array_buffer,
            generalized_suffix_array: self.generalized_suffix_array,
            plcp_buffer: self.plcp_buffer,
            thread_count: self.thread_count,
            _parallelism_marker: PhantomData,
            _buffer_mode_marker: PhantomData,
        }
    }
}

#[cfg(feature = "openmp")]
impl<'p, 's, 't, I: InputElement, O: OutputElement, B: BufferMode>
    PlcpConstruction<'p, 's, 't, I, O, B, SingleThreaded>
{
    pub fn multithreaded(
        mut self,
        thread_count: ThreadCount,
    ) -> PlcpConstruction<'p, 's, 't, I, O, B, MultiThreaded> {
        self.thread_count = thread_count;
        self.into_other_marker_type()
    }
}

impl<'p, 's, 't, I: InputElement, O: OutputElement, P: Parallelism>
    PlcpConstruction<'p, 's, 't, I, O, OwnedBuffer, P>
{
    pub fn with_borrowed_buffer(
        mut self,
        plcp_buffer: &'p mut [O],
    ) -> PlcpConstruction<'p, 's, 't, I, O, BorrowedBuffer, P> {
        self.plcp_buffer = Some(plcp_buffer);
        self.into_other_marker_type()
    }
}

impl<'p, 's, 't, I: InputElement, O: OutputElement, P: Parallelism>
    PlcpConstruction<'p, 's, 't, I, O, OwnedBuffer, P>
{
    pub fn construct(self) -> Result<PlcpWithBorrowedSuffixArray<'s, O>, SaisError> {
        let mut plcp_buffer = vec![O::try_from(0).unwrap(); self.text.len()];

        let mut construction = self.into_other_marker_type::<BorrowedBuffer, P>();
        construction.plcp_buffer = Some(&mut plcp_buffer);

        let res = construction.construct_in_borrowed_buffer();

        match res {
            Ok(borrowed) => {
                let (_, suffix_array) = borrowed.into_parts();

                Ok(PlcpWithBorrowedSuffixArray {
                    plcp: plcp_buffer,
                    suffix_array,
                })
            }
            Err(e) => Err(e),
        }
    }
}

impl<'p, 's, 't, I: InputElement, O: OutputElement, P: Parallelism>
    PlcpConstruction<'p, 's, 't, I, O, BorrowedBuffer, P>
{
    pub fn construct_in_borrowed_buffer(
        mut self,
    ) -> Result<BorrowedPlcpWithBorrowedSuffixArray<'p, 's, O>, SaisError> {
        let plcp_buffer = self.plcp_buffer.take().unwrap();

        assert_eq!(self.text.len(), plcp_buffer.len());
        assert_eq!(self.text.len(), self.suffix_array_buffer.len());

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

        unsafe {
            <<P::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<I,O>>::LcpFunctions::run_libsais_plcp(
                self.text.as_ptr(),
                self.suffix_array_buffer.as_ptr(),
                plcp_buffer.as_mut_ptr(),
                text_len,
                num_threads,
                self.generalized_suffix_array
            )
        }.into_empty_sais_result().map(|_| BorrowedPlcpWithBorrowedSuffixArray {
            plcp: plcp_buffer,
            suffix_array: self.suffix_array_buffer,
        })
    }
}
