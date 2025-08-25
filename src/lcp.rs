use std::marker::PhantomData;

use crate::{
    OutputElement, ThreadCount,
    error::{IntoSaisResult, LibsaisError},
    generics_dispatch::{LcpFunctionsDispatch, LibsaisLcpFunctions},
    owned_or_borrowed::OwnedOrBorrowed,
    type_state::{
        BorrowedBuffer, BufferMode, BufferModeOrReplaceInput, OwnedBuffer, Parallelism,
        ParallelismOrUndecided, ReplaceInput,
    },
};

#[allow(unused)]
use crate::type_state::SingleThreaded;

#[cfg(feature = "openmp")]
use crate::type_state::{MultiThreaded, Undecided};

#[derive(Debug)]
pub struct LcpConstruction<
    'l,
    'p,
    's,
    O: OutputElement,
    LcpB: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
    SaB: BufferMode,
    P: ParallelismOrUndecided,
> {
    pub(crate) plcp_buffer: OwnedOrBorrowed<'p, O, PlcpB>,
    pub(crate) suffix_array_buffer: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) lcp_buffer: Option<&'l mut [O]>,
    pub(crate) thread_count: ThreadCount,
    pub(crate) is_generalized_suffix_array: bool,
    pub(crate) _parallelism_marker: PhantomData<P>,
    pub(crate) _lcp_buffer_mode_marker: PhantomData<LcpB>,
}

impl<
    'l,
    'p,
    's,
    O: OutputElement,
    LcpB1: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
    SaB: BufferMode,
    P1: ParallelismOrUndecided,
> LcpConstruction<'l, 'p, 's, O, LcpB1, PlcpB, SaB, P1>
{
    fn into_other_marker_type<LcpB2: BufferModeOrReplaceInput, P2: ParallelismOrUndecided>(
        self,
    ) -> LcpConstruction<'l, 'p, 's, O, LcpB2, PlcpB, SaB, P2> {
        LcpConstruction {
            plcp_buffer: self.plcp_buffer,
            suffix_array_buffer: self.suffix_array_buffer,
            lcp_buffer: self.lcp_buffer,
            thread_count: self.thread_count,
            is_generalized_suffix_array: self.is_generalized_suffix_array,
            _parallelism_marker: PhantomData,
            _lcp_buffer_mode_marker: PhantomData,
        }
    }
}

impl<'p, 's, O: OutputElement, PlcpB: BufferMode, SaB: BufferMode, P: ParallelismOrUndecided>
    LcpConstruction<'static, 'p, 's, O, OwnedBuffer, PlcpB, SaB, P>
{
    pub fn in_borrowed_buffer<'l>(
        self,
        lcp_buffer: &'l mut [O],
    ) -> LcpConstruction<'l, 'p, 's, O, BorrowedBuffer, PlcpB, SaB, P> {
        LcpConstruction {
            plcp_buffer: self.plcp_buffer,
            suffix_array_buffer: self.suffix_array_buffer,
            lcp_buffer: Some(lcp_buffer),
            thread_count: self.thread_count,
            is_generalized_suffix_array: self.is_generalized_suffix_array,
            _parallelism_marker: PhantomData,
            _lcp_buffer_mode_marker: PhantomData,
        }
    }

    pub fn replace_suffix_array(
        self,
    ) -> LcpConstruction<'static, 'p, 's, O, ReplaceInput, PlcpB, SaB, P> {
        self.into_other_marker_type()
    }
}

impl<
    'l,
    'p,
    's,
    O: OutputElement,
    LcpB: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
    SaB: BufferMode,
> LcpConstruction<'l, 'p, 's, O, LcpB, PlcpB, SaB, Undecided>
{
    pub fn single_threaded(
        self,
    ) -> LcpConstruction<'l, 'p, 's, O, LcpB, PlcpB, SaB, SingleThreaded> {
        self.into_other_marker_type()
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        mut self,
        thread_count: ThreadCount,
    ) -> LcpConstruction<'l, 'p, 's, O, LcpB, PlcpB, SaB, MultiThreaded> {
        self.thread_count = thread_count;
        self.into_other_marker_type()
    }
}

impl<
    'l,
    'p,
    's,
    O: OutputElement,
    LcpB: BufferMode,
    PlcpB: BufferMode,
    SaB: BufferMode,
    P: Parallelism,
> LcpConstruction<'l, 'p, 's, O, LcpB, PlcpB, SaB, P>
{
    pub fn run(
        mut self,
    ) -> Result<SuffixArrayWithLcpAndPlcp<'l, 'p, 's, O, LcpB, PlcpB, SaB>, LibsaisError> {
        let mut lcp = OwnedOrBorrowed::take_buffer_or_allocate(self.lcp_buffer.take(), || {
            vec![O::ZERO; self.suffix_array_buffer.buffer.len()]
        });

        self.run_in_optional_borrowed_buffer(Some(&mut lcp.buffer))
            .map(|_| SuffixArrayWithLcpAndPlcp {
                lcp,
                plcp: self.plcp_buffer,
                suffix_array: self.suffix_array_buffer,
                is_generalized_suffix_array: self.is_generalized_suffix_array,
            })
    }
}

impl<'l, 'p, 's, O: OutputElement, PlcpB: BufferMode, SaB: BufferMode, P: Parallelism>
    LcpConstruction<'l, 'p, 's, O, ReplaceInput, PlcpB, SaB, P>
{
    pub fn run(mut self) -> Result<LcpAndPlcp<'s, 'p, O, SaB, PlcpB>, LibsaisError> {
        self.run_in_optional_borrowed_buffer(None)
            .map(|_| LcpAndPlcp {
                lcp: self.suffix_array_buffer,
                plcp: self.plcp_buffer,
            })
    }
}

impl<
    'l,
    'p,
    's,
    O: OutputElement,
    LcpB: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
    SaB: BufferMode,
    P: Parallelism,
> LcpConstruction<'l, 'p, 's, O, LcpB, PlcpB, SaB, P>
{
    fn run_in_optional_borrowed_buffer(
        &mut self,
        lcp_buffer_opt: Option<&mut [O]>,
    ) -> Result<(), LibsaisError> {
        assert_eq!(
            self.suffix_array_buffer.buffer.len(),
            self.plcp_buffer.buffer.len()
        );
        if let Some(lcp_buffer) = lcp_buffer_opt.as_ref() {
            assert_eq!(self.suffix_array_buffer.buffer.len(), lcp_buffer.len());
        }

        // the try_into implementations fail exactly when the value is too large for the respective libsais version
        let Ok(suffix_array_len): Result<O, _> = self.suffix_array_buffer.buffer.len().try_into()
        else {
            panic!(
                "The suffix array is too long for the chosen output type. Suffix array len: {}, Max allowed len: {}",
                self.suffix_array_buffer.buffer.len(),
                O::MAX
            );
        };

        let num_threads = O::try_from(self.thread_count.value as usize).unwrap();

        // this and the below break Rust's borrowing rules for suffix_array_buffer
        // but the pointers are only used in the C code
        let lcp_ptr = lcp_buffer_opt.map_or_else(
            || self.suffix_array_buffer.buffer.as_mut_ptr(),
            |lcp_buffer| lcp_buffer.as_mut_ptr(),
        );

        // SAFETY: lens of buffers were checked
        // the content of the buffer must be correct, because this object could only be attained by
        // either claiming so in an unsafe fn or by constructing them using the appropriate functions of
        // this library
        unsafe {
            LcpFunctionsDispatch::<u8, O, P>::libsais_lcp(
                self.plcp_buffer.buffer.as_ptr(),
                self.suffix_array_buffer.buffer.as_ptr(),
                lcp_ptr,
                suffix_array_len,
                num_threads,
            )
        }
        .into_empty_sais_result()
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SuffixArrayWithLcpAndPlcp<
    'l,
    'p,
    's,
    O: OutputElement,
    LcpB: BufferMode,
    PlcpB: BufferMode,
    SaB: BufferMode,
> {
    pub(crate) lcp: OwnedOrBorrowed<'l, O, LcpB>,
    pub(crate) plcp: OwnedOrBorrowed<'p, O, PlcpB>,
    pub(crate) suffix_array: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'l, 'p, 's, O: OutputElement, LcpB: BufferMode, PlcpB: BufferMode, SaB: BufferMode>
    SuffixArrayWithLcpAndPlcp<'l, 'p, 's, O, LcpB, PlcpB, SaB>
{
    pub fn lcp(&self) -> &[O] {
        &self.lcp.buffer
    }

    pub fn plcp(&self) -> &[O] {
        &self.plcp.buffer
    }

    pub fn suffix_array(&self) -> &[O] {
        &self.suffix_array.buffer
    }

    pub fn is_generalized_suffix_array(&self) -> bool {
        self.is_generalized_suffix_array
    }

    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        SaB::Buffer<'s, O>,
        LcpB::Buffer<'l, O>,
        PlcpB::Buffer<'p, O>,
        bool,
    ) {
        (
            self.suffix_array.into_inner(),
            self.lcp.into_inner(),
            self.plcp.into_inner(),
            self.is_generalized_suffix_array,
        )
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct LcpAndPlcp<'l, 'p, O: OutputElement, LcpB: BufferMode, PlcpB: BufferMode> {
    pub(crate) lcp: OwnedOrBorrowed<'l, O, LcpB>,
    pub(crate) plcp: OwnedOrBorrowed<'p, O, PlcpB>,
}

impl<'l, 'p, O: OutputElement, LcpB: BufferMode, PlcpB: BufferMode>
    LcpAndPlcp<'l, 'p, O, LcpB, PlcpB>
{
    pub fn lcp(&self) -> &[O] {
        &self.lcp.buffer
    }

    pub fn plcp(&self) -> &[O] {
        &self.plcp.buffer
    }

    pub fn into_parts(self) -> (LcpB::Buffer<'l, O>, PlcpB::Buffer<'p, O>) {
        (self.lcp.into_inner(), self.plcp.into_inner())
    }
}
