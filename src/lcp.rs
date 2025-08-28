/*!
 * Construct the [longest common prefix array] (LCP) from a permuted LCP array (PLCP) for a suffix array.
 *
 * [`LcpConstruction`] provides a builder-like API for constructing the PLCP. It can only be obtained from
 * a [`SuffixArrayWitPlcp`](super::plcp::SuffixArrayWithPlcp), which is in turn obtained from a
 * PLCP array construction or by using its `unsafe` constructor.
 *
 * A notable configuration option is the ability to replace the _suffix array_ (not the PLCP array) with
 * the LCP array.
 *
 * ```
 * use libsais::{SuffixArrayConstruction};
 *
 * let text = b"abracadabra".as_slice();
 *
 * let res = SuffixArrayConstruction::for_text(text)
 *     .in_owned_buffer32()
 *     .single_threaded()
 *     .run()
 *     .unwrap();
 *
 * let res_with_plcp = res.plcp_construction()
 *     .single_threaded()
 *     .run()
 *     .unwrap();
 *
 * let res_with_plcp_and_lcp = res_with_plcp.lcp_construction()
 *     .single_threaded()
 *     .run()
 *     .unwrap();
 * ```
 *
 * # Output Convention
 *
 * The LCP array of `libsais` always starts with a 0. The second entry is the LCP value for the first two suffixes
 * of the suffix array, and so on.
 *
 * # Generalized Suffix Array Support
 *
 * When using the generalized suffix array mode, the longest common prefix calculation behaves as theoretically
 * expected. Only the prefixes of individual texts are compared and the sentinels stop the comparison.
 *
 * [longest common prefix array]: https://en.wikipedia.org/wiki/LCP_array
 */

use std::marker::PhantomData;

use crate::{
    IntoSaisResult, LibsaisError, OutputElement, ThreadCount,
    generics_dispatch::{LcpFunctionsDispatch, LibsaisLcpFunctions},
    owned_or_borrowed::OwnedOrBorrowed,
    typestate::{
        BorrowedBuffer, BufferMode, BufferModeOrReplaceInput, OwnedBuffer, Parallelism,
        ParallelismOrUndecided, ReplaceInput,
    },
};

#[allow(unused)]
use crate::typestate::SingleThreaded;

#[cfg(feature = "openmp")]
use crate::typestate::{MultiThreaded, Undecided};

/// Construct the permuted longest common prefix array for a suffix array and PLCP.
///
/// See [`lcp`](self) for details.
#[derive(Debug)]
pub struct LcpConstruction<
    'l,
    'p,
    's,
    O: OutputElement,
    SaB: BufferMode,
    LcpB: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
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
    SaB: BufferMode,
    LcpB1: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
    P1: ParallelismOrUndecided,
> LcpConstruction<'l, 'p, 's, O, SaB, LcpB1, PlcpB, P1>
{
    fn into_other_marker_type<LcpB2: BufferModeOrReplaceInput, P2: ParallelismOrUndecided>(
        self,
    ) -> LcpConstruction<'l, 'p, 's, O, SaB, LcpB2, PlcpB, P2> {
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

impl<'p, 's, O: OutputElement, SaB: BufferMode, PlcpB: BufferMode, P: ParallelismOrUndecided>
    LcpConstruction<'static, 'p, 's, O, SaB, OwnedBuffer, PlcpB, P>
{
    /// Construct the LCP array in a borrowed buffer instead of allocating an owned [`Vec`].
    ///
    /// The buffer must have the same length as the suffix array and PLCP.
    pub fn in_borrowed_buffer<'l>(
        self,
        lcp_buffer: &'l mut [O],
    ) -> LcpConstruction<'l, 'p, 's, O, SaB, BorrowedBuffer, PlcpB, P> {
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
    ) -> LcpConstruction<'static, 'p, 's, O, SaB, ReplaceInput, PlcpB, P> {
        self.into_other_marker_type()
    }
}

impl<
    'l,
    'p,
    's,
    O: OutputElement,
    SaB: BufferMode,
    LcpB: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
> LcpConstruction<'l, 'p, 's, O, SaB, LcpB, PlcpB, Undecided>
{
    pub fn single_threaded(
        self,
    ) -> LcpConstruction<'l, 'p, 's, O, SaB, LcpB, PlcpB, SingleThreaded> {
        self.into_other_marker_type()
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        mut self,
        thread_count: ThreadCount,
    ) -> LcpConstruction<'l, 'p, 's, O, SaB, LcpB, PlcpB, MultiThreaded> {
        self.thread_count = thread_count;
        self.into_other_marker_type()
    }
}

impl<
    'l,
    'p,
    's,
    O: OutputElement,
    SaB: BufferMode,
    LcpB: BufferMode,
    PlcpB: BufferMode,
    P: Parallelism,
> LcpConstruction<'l, 'p, 's, O, SaB, LcpB, PlcpB, P>
{
    pub fn run(
        mut self,
    ) -> Result<SuffixArrayWithLcpAndPlcp<'l, 'p, 's, O, SaB, LcpB, PlcpB>, LibsaisError> {
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

impl<'l, 'p, 's, O: OutputElement, SaB: BufferMode, PlcpB: BufferMode, P: Parallelism>
    LcpConstruction<'l, 'p, 's, O, SaB, ReplaceInput, PlcpB, P>
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
    SaB: BufferMode,
    LcpB: BufferModeOrReplaceInput,
    PlcpB: BufferMode,
    P: Parallelism,
> LcpConstruction<'l, 'p, 's, O, SaB, LcpB, PlcpB, P>
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

/// The read-only return type of an LCP construction.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SuffixArrayWithLcpAndPlcp<
    'l,
    'p,
    's,
    O: OutputElement,
    SaB: BufferMode,
    LcpB: BufferMode,
    PlcpB: BufferMode,
> {
    pub(crate) lcp: OwnedOrBorrowed<'l, O, LcpB>,
    pub(crate) plcp: OwnedOrBorrowed<'p, O, PlcpB>,
    pub(crate) suffix_array: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'l, 'p, 's, O: OutputElement, SaB: BufferMode, LcpB: BufferMode, PlcpB: BufferMode>
    SuffixArrayWithLcpAndPlcp<'l, 'p, 's, O, SaB, LcpB, PlcpB>
{
    pub fn suffix_array(&self) -> &[O] {
        &self.suffix_array.buffer
    }

    pub fn lcp(&self) -> &[O] {
        &self.lcp.buffer
    }

    pub fn plcp(&self) -> &[O] {
        &self.plcp.buffer
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

/// The read-only return type of an LCP construction, when the LCP has replaced the suffix array.
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
