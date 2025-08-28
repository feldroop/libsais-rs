/*!
 * Construct the [permuted longest common prefix array]
 * (PLCP) for a suffix array and text.
 *
 * The PLCP is a similar data structure to the longest common prefix array (LCP). The difference is that
 * the longest common prefix values appear in text order rather than lexicographical order.
 * It can be defined as follows: `PLCP[SUF[j]] = p <=> LCP[j] = p`.
 *
 * A PLCP can be used to simulate or construct the LCP array. Please refer to the [literature]
 * for details. This library only supports constructing the LCP array, see [`lcp`](super::lcp) for details.
 *
 * [`PlcpConstruction`] provides a builder-like API for constructing the PLCP. It can only be obtained from
 * a [`SuffixArrayWithText`](super::suffix_array::SuffixArrayWithText), which is in turn obtained from a
 * suffix array construction or by using its `unsafe` constructor.
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
 * ```
 *
 * # Generalized Suffix Array Support
 *
 * When using the generalized suffix array mode, the longest common prefix calculation behaves as theoretically
 * expected. Only the prefixes of individual texts are compared and the sentinels stop the comparison.
 *
 * [permuted longest common prefix array]: https://doi.org/10.1007/978-3-642-02441-2_17
 * [literature]: https://doi.org/10.1007/978-3-642-02441-2_17
 */

use std::marker::PhantomData;

use crate::{
    InputElement, IntoSaisResult, LibsaisError, OutputElement, ThreadCount,
    generics_dispatch::{LcpFunctionsDispatch, LibsaisLcpFunctions},
    lcp::LcpConstruction,
    owned_or_borrowed::OwnedOrBorrowed,
    typestate::{
        BorrowedBuffer, BufferMode, OwnedBuffer, Parallelism, ParallelismOrUndecided, Undecided,
    },
};

#[allow(unused)]
use crate::typestate::SingleThreaded;

#[cfg(feature = "openmp")]
use crate::typestate::MultiThreaded;

/// Construct the permuted longest common prefix array for a suffix array and text.
///
/// See [`plcp`](self) for details.
#[derive(Debug)]
pub struct PlcpConstruction<
    'p,
    's,
    't,
    I: InputElement,
    O: OutputElement,
    SaB: BufferMode,
    PlcpB: BufferMode,
    P: ParallelismOrUndecided,
> {
    pub(crate) text: &'t [I],
    pub(crate) suffix_array_buffer: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) generalized_suffix_array: bool,
    pub(crate) plcp_buffer: Option<&'p mut [O]>,
    pub(crate) thread_count: ThreadCount,
    pub(crate) _parallelism_marker: PhantomData<P>,
    pub(crate) _plcp_buffer_mode_marker: PhantomData<PlcpB>,
}

impl<'p, 's, 't, I: InputElement, O: OutputElement, SaB: BufferMode, PlcpB: BufferMode>
    PlcpConstruction<'p, 's, 't, I, O, SaB, PlcpB, Undecided>
{
    pub fn single_threaded(self) -> PlcpConstruction<'p, 's, 't, I, O, SaB, PlcpB, SingleThreaded> {
        PlcpConstruction {
            text: self.text,
            suffix_array_buffer: self.suffix_array_buffer,
            generalized_suffix_array: self.generalized_suffix_array,
            plcp_buffer: self.plcp_buffer,
            thread_count: ThreadCount::fixed(1),
            _parallelism_marker: PhantomData,
            _plcp_buffer_mode_marker: PhantomData,
        }
    }

    #[cfg(feature = "openmp")]
    pub fn multi_threaded(
        self,
        thread_count: ThreadCount,
    ) -> PlcpConstruction<'p, 's, 't, I, O, SaB, PlcpB, MultiThreaded> {
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

impl<'s, 't, I: InputElement, O: OutputElement, SaB: BufferMode, P: ParallelismOrUndecided>
    PlcpConstruction<'static, 's, 't, I, O, SaB, OwnedBuffer, P>
{
    /// Construct the PLCP array in a borrowed buffer instead of allocating an owned [`Vec`].
    ///
    /// The buffer must have the same length as the suffix array and text.
    pub fn in_borrowed_buffer<'p>(
        self,
        plcp_buffer: &'p mut [O],
    ) -> PlcpConstruction<'p, 's, 't, I, O, SaB, BorrowedBuffer, P> {
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
    SaB: BufferMode,
    PlcpB: BufferMode,
    P: Parallelism,
> PlcpConstruction<'p, 's, 't, I, O, SaB, PlcpB, P>
{
    /// Construct the PLCP array for the given suffix array and text.
    ///
    /// # Panics
    ///
    /// If the text, suffix array and PLCP buffers don't all have the same length, which has to fit into
    /// the output type. When using a generalized suffix array, the last character of the text has to be the
    /// zero byte (not ASCII '0').
    ///
    /// # Returns
    ///
    /// An error or a type that bundles the suffix array with the PLCP array.
    pub fn run(mut self) -> Result<SuffixArrayWithPlcp<'p, 's, O, SaB, PlcpB>, LibsaisError> {
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

    fn construct_in_buffer(&mut self, plcp_buffer: &mut [O]) -> Result<(), LibsaisError> {
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
                (*c).into() == 0i64,
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

/// The read-only return type of a PLCP construction.
///
/// It bundles the suffix array and the PLCP array, to safely allow LCP array construction.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SuffixArrayWithPlcp<'p, 's, O: OutputElement, SaB: BufferMode, PlcpB: BufferMode> {
    pub(crate) suffix_array: OwnedOrBorrowed<'s, O, SaB>,
    pub(crate) plcp: OwnedOrBorrowed<'p, O, PlcpB>,
    pub(crate) is_generalized_suffix_array: bool,
}

impl<'p, 's, O: OutputElement, SaB: BufferMode, PlcpB: BufferMode>
    SuffixArrayWithPlcp<'p, 's, O, SaB, PlcpB>
{
    pub fn suffix_array(&self) -> &[O] {
        &self.suffix_array.buffer
    }

    pub fn plcp(&self) -> &[O] {
        &self.plcp.buffer
    }

    pub fn is_generalized_suffix_array(&self) -> bool {
        self.is_generalized_suffix_array
    }

    pub fn into_parts(self) -> (SaB::Buffer<'s, O>, PlcpB::Buffer<'p, O>, bool) {
        (
            self.suffix_array.into_inner(),
            self.plcp.into_inner(),
            self.is_generalized_suffix_array,
        )
    }

    /// Construct this type without going through a [`PlcpConstruction`] or by using the parts
    /// obtained by [`Self::into_parts`].
    ///
    /// # Safety
    ///
    /// You are claiming that the PLCP array is correct for the suffix array according to the conventions of `libsais`
    /// and that the indicator for the generalized suffix array is correct.
    pub unsafe fn from_parts(
        plcp: PlcpB::Buffer<'p, O>,
        suffix_array: SaB::Buffer<'s, O>,
        is_generalized_suffix_array: bool,
    ) -> Self {
        Self {
            plcp: OwnedOrBorrowed::new(plcp),
            suffix_array: OwnedOrBorrowed::new(suffix_array),
            is_generalized_suffix_array,
        }
    }

    pub fn lcp_construction(
        self,
    ) -> LcpConstruction<'static, 'p, 's, O, SaB, OwnedBuffer, PlcpB, Undecided> {
        LcpConstruction {
            plcp_buffer: self.plcp,
            suffix_array_buffer: self.suffix_array,
            lcp_buffer: None,
            thread_count: ThreadCount::fixed(1),
            is_generalized_suffix_array: self.is_generalized_suffix_array,
            _parallelism_marker: PhantomData,
            _lcp_buffer_mode_marker: PhantomData,
        }
    }
}
