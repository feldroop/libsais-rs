use std::{ffi::c_void, marker::PhantomData};

use crate::{
    InputElement, OutputElement, SmallAlphabet, ThreadCount,
    generics_dispatch::{LibsaisFunctionsSmallAlphabet, SmallAlphabetFunctionsDispatch},
    type_state::{OutputElementOrUndecided, Parallelism, ParallelismOrUndecided, SingleThreaded},
};

#[cfg(feature = "openmp")]
use crate::type_state::MultiThreaded;

#[derive(Debug, PartialEq, Eq)]
pub struct Context<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> {
    ptr: *mut c_void,
    num_threads: u16,
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
    _parallelism_marker: PhantomData<P>,
}

impl<I: SmallAlphabet> Context<I, i32, SingleThreaded> {
    pub fn new_single_threaded() -> Self {
        Self::new(ThreadCount::fixed(1))
    }

    pub fn try_new_single_threaded() -> Result<Self, ()> {
        Self::try_new(ThreadCount::fixed(1))
    }
}

#[cfg(feature = "openmp")]
impl<I: SmallAlphabet> Context<I, i32, MultiThreaded> {
    pub fn new_multi_threaded(thread_count: ThreadCount) -> Self {
        Self::new(thread_count)
    }

    pub fn try_new_multi_threaded(thread_count: ThreadCount) -> Result<Self, ()> {
        Self::try_new(thread_count)
    }
}

impl<I: SmallAlphabet, P: Parallelism> Context<I, i32, P> {
    fn new(thread_count: ThreadCount) -> Self {
        Self::try_new(thread_count).expect("libsais create ctx should not return nullpointer")
    }

    fn try_new(thread_count: ThreadCount) -> Result<Self, ()> {
        // SAFETY: constructing the context is not not unsafe
        let ptr = unsafe {
            SmallAlphabetFunctionsDispatch::<I, i32, P>::libsais_create_ctx(
                thread_count.value as i32,
            )
        };

        if ptr.is_null() {
            Err(())
        } else {
            Ok(Self {
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
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn num_threads(&self) -> u16 {
        self.num_threads
    }
}

impl<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> Drop
    for Context<I, O, P>
{
    fn drop(&mut self) {
        // TODO
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        //unsafe { SmallAlphabetFunctionsDispatch::<I, O, P>::libsais_free_ctx(self.ptr) }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct UnBwtContext<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> {
    ptr: *mut c_void,
    num_threads: u16,
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
    _parallelism_marker: PhantomData<P>,
}

impl<I: SmallAlphabet> UnBwtContext<I, i32, SingleThreaded> {
    pub fn new_single_threaded() -> Self {
        Self::new(ThreadCount::fixed(1))
    }

    pub fn try_new_single_threaded() -> Result<Self, ()> {
        Self::try_new(ThreadCount::fixed(1))
    }
}

#[cfg(feature = "openmp")]
impl<I: SmallAlphabet> UnBwtContext<I, i32, MultiThreaded> {
    pub fn new_multi_threaded(thread_count: ThreadCount) -> Self {
        Self::new(thread_count)
    }

    pub fn try_new_multi_threaded(thread_count: ThreadCount) -> Result<Self, ()> {
        Self::try_new(thread_count)
    }
}

impl<I: SmallAlphabet, P: Parallelism> UnBwtContext<I, i32, P> {
    fn new(thread_count: ThreadCount) -> Self {
        Self::try_new(thread_count).expect("libsais create ctx should not return nullpointer")
    }

    fn try_new(thread_count: ThreadCount) -> Result<Self, ()> {
        // SAFETY: constructing the context is not not unsafe
        let ptr = unsafe {
            SmallAlphabetFunctionsDispatch::<I, i32, P>::libsais_unbwt_create_ctx(
                thread_count.value as i32,
            )
        };

        if ptr.is_null() {
            Err(())
        } else {
            Ok(Self {
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
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn num_threads(&self) -> u16 {
        self.num_threads
    }
}

impl<I: InputElement, O: OutputElementOrUndecided, P: ParallelismOrUndecided> Drop
    for UnBwtContext<I, O, P>
{
    fn drop(&mut self) {
        // TODO
        // SAFETY: this pointer was acquired by calling one of the corresponding unbwt_create_ctx functions
        // unsafe { SmallAlphabetFunctionsDispatch::<P, I, O>::libsais_unbwt_free_ctx(self.ptr) }
    }
}
