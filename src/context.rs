/*!
 * Use an optimization for repeated calls of `libsais` functions on small inputs.
 *
 * Using a context allowes `libsais` to reuse memory allocations. This is a small optimization
 * that is only relevant for applications that repeatedly call `libsais` functions on small
 * inputs (<64K).
 *
 * The use of contexts is only available for `u8`/`u16`-based input texts.
 */

use std::{ffi::c_void, marker::PhantomData};

use crate::{
    InputElement, OutputElement, SmallAlphabet, ThreadCount,
    generics_dispatch::{
        LibsaisFunctionsSmallAlphabet, SmallAlphabetFunctionsDispatch,
        SmallAlphabetFunctionsDispatchOrUnimplemented,
    },
    typestate::{OutputElementOrUndecided, Parallelism, ParallelismOrUndecided, SingleThreaded},
};

#[cfg(feature = "openmp")]
use crate::typestate::MultiThreaded;

/// A context for use in suffix array and BWT construction. Refer to [`context`](self) for details.
#[derive(Debug, PartialEq, Eq)]
pub struct Context<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> {
    ptr: *mut c_void,
    num_threads: u16,
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
    _parallelism_marker: PhantomData<P>,
}

impl<I: SmallAlphabet> Context<I, i32, SingleThreaded> {
    /// Create a context for use in single threaded operations.
    ///
    /// # Panics
    ///
    /// If `libsais` returns the nullpointer. Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn new_single_threaded() -> Self {
        Self::new(ThreadCount::fixed(1))
    }

    /// Create a context for use in single threaded operations.
    ///
    /// Returns [None] if `libsais` returns the nullpointer.
    /// Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn try_new_single_threaded() -> Option<Self> {
        Self::try_new(ThreadCount::fixed(1))
    }
}

#[cfg(feature = "openmp")]
impl<I: SmallAlphabet> Context<I, i32, MultiThreaded> {
    /// Create a context for use in multi threaded operations.
    ///
    /// # Arguments
    ///
    /// * `thread_count` - The number of threads to use. This MUST be the same value as is used in
    ///   the configuration of the algorithms in the builder API. Otherwise the builder will panic.
    ///
    /// # Panics
    ///
    /// If `libsais` returns the nullpointer. Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn new_multi_threaded(thread_count: ThreadCount) -> Self {
        Self::new(thread_count)
    }

    /// Create a context for use in multi threaded operations.
    ///
    /// # Arguments
    ///
    /// * `thread_count` - The number of threads to use. This MUST be the same value as is used in
    ///   the configuration of the algorithms in the builder API. Otherwise the builder will panic.
    ///
    /// Returns [`None`] if `libsais` returns the nullpointer.
    /// Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn try_new_multi_threaded(thread_count: ThreadCount) -> Option<Self> {
        Self::try_new(thread_count)
    }
}

impl<I: SmallAlphabet, P: Parallelism> Context<I, i32, P> {
    fn new(thread_count: ThreadCount) -> Self {
        Self::try_new(thread_count).expect("libsais create ctx should not return nullpointer")
    }

    fn try_new(thread_count: ThreadCount) -> Option<Self> {
        // SAFETY: constructing the context is not not unsafe
        let ptr = unsafe {
            SmallAlphabetFunctionsDispatch::<I, i32, P>::libsais_create_ctx(
                thread_count.value as i32,
            )
        };

        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                num_threads: thread_count.value,
                _input_marker: PhantomData,
                _output_marker: PhantomData,
                _parallelism_marker: PhantomData,
            })
        }
    }
}

impl<I: InputElement, O: OutputElement, P: Parallelism> Context<I, O, P> {
    pub(crate) fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Returns the configured number of threads.
    pub fn num_threads(&self) -> u16 {
        self.num_threads
    }
}

impl<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> Drop
    for Context<I, O, P>
{
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        unsafe {
            SmallAlphabetFunctionsDispatchOrUnimplemented::<I, O, P>::libsais_free_ctx(self.ptr)
        }
    }
}

/// A context for use in the recovering of texts from BWTs. Refer to [`context`](self) for details.
#[derive(Debug, PartialEq, Eq)]
pub struct UnBwtContext<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> {
    ptr: *mut c_void,
    num_threads: u16,
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
    _parallelism_marker: PhantomData<P>,
}

impl<I: SmallAlphabet> UnBwtContext<I, i32, SingleThreaded> {
    /// Create a context for use in single threaded operations.
    ///
    /// # Panics
    ///
    /// If `libsais` returns the nullpointer. Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn new_single_threaded() -> Self {
        Self::new(ThreadCount::fixed(1))
    }

    /// Create a context for use in single threaded operations.
    ///
    /// Returns [`None`] if `libsais` returns the nullpointer.
    /// Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn try_new_single_threaded() -> Option<Self> {
        Self::try_new(ThreadCount::fixed(1))
    }
}

#[cfg(feature = "openmp")]
impl<I: SmallAlphabet> UnBwtContext<I, i32, MultiThreaded> {
    /// Create a context for use in multi threaded operations.
    ///
    /// # Arguments
    ///
    /// * `thread_count` - The number of threads to use. This MUST be the same value as is used in
    ///   the configuration of the algorithms in the builder API. Otherwise the builder will panic.
    ///
    /// # Panics
    ///
    /// If `libsais` returns the nullpointer. Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn new_multi_threaded(thread_count: ThreadCount) -> Self {
        Self::new(thread_count)
    }

    /// Create a context for use in multi threaded operations.
    ///
    /// # Arguments
    ///
    /// * `thread_count` - The number of threads to use. This MUST be the same value as is used in
    ///   the configuration of the algorithms in the builder API. Otherwise the builder will panic.
    ///
    /// Returns [`None`] if `libsais` returns the nullpointer.
    /// Could be due to out-of-memory issues. Is unlikely to happen.
    pub fn try_new_multi_threaded(thread_count: ThreadCount) -> Option<Self> {
        Self::try_new(thread_count)
    }
}

impl<I: SmallAlphabet, P: Parallelism> UnBwtContext<I, i32, P> {
    fn new(thread_count: ThreadCount) -> Self {
        Self::try_new(thread_count).expect("libsais create ctx should not return nullpointer")
    }

    fn try_new(thread_count: ThreadCount) -> Option<Self> {
        // SAFETY: constructing the context is not not unsafe
        let ptr = unsafe {
            SmallAlphabetFunctionsDispatch::<I, i32, P>::libsais_unbwt_create_ctx(
                thread_count.value as i32,
            )
        };

        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                num_threads: thread_count.value,
                _input_marker: PhantomData,
                _output_marker: PhantomData,
                _parallelism_marker: PhantomData,
            })
        }
    }
}

impl<I: InputElement, O: OutputElement, P: Parallelism> UnBwtContext<I, O, P> {
    pub(crate) fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Returns the configured number of threads.
    pub fn num_threads(&self) -> u16 {
        self.num_threads
    }
}

impl<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> Drop
    for UnBwtContext<I, O, P>
{
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding unbwt_create_ctx functions
        unsafe {
            SmallAlphabetFunctionsDispatchOrUnimplemented::<I, O, P>::libsais_unbwt_free_ctx(
                self.ptr,
            )
        }
    }
}
