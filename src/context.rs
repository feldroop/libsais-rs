use libsais_sys::{libsais, libsais16};

use std::ffi::c_void;

mod sealed {
    pub trait Sealed {}
}

// TODO unbwt contexts, libsais16 contexts

pub trait SaisContext: sealed::Sealed {
    fn as_mut_ptr(&mut self) -> *mut c_void;

    fn num_threads(&self) -> u16;
}

pub struct SingleThreaded8InputSaisContext {
    pub(crate) ptr: *mut c_void,
}

impl SingleThreaded8InputSaisContext {
    pub fn new() -> Self {
        // SAFETY: constructing the context is not not unsafe
        Self {
            ptr: unsafe { libsais::libsais_create_ctx() },
        }
    }
}

impl sealed::Sealed for SingleThreaded8InputSaisContext {}

impl SaisContext for SingleThreaded8InputSaisContext {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn num_threads(&self) -> u16 {
        1
    }
}

impl Drop for SingleThreaded8InputSaisContext {
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        unsafe {
            libsais::libsais_free_ctx(self.ptr);
        }
    }
}

pub struct SingleThreaded16InputSaisContext {
    pub(crate) ptr: *mut c_void,
}

impl SingleThreaded16InputSaisContext {
    pub fn new() -> Self {
        // SAFETY: constructing the context is not not unsafe
        Self {
            ptr: unsafe { libsais16::libsais16_create_ctx() },
        }
    }
}

impl sealed::Sealed for SingleThreaded16InputSaisContext {}

impl SaisContext for SingleThreaded16InputSaisContext {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn num_threads(&self) -> u16 {
        1
    }
}

impl Drop for SingleThreaded16InputSaisContext {
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        unsafe {
            libsais16::libsais16_free_ctx(self.ptr);
        }
    }
}

/// There is currently no use for the multithreaded 8-bit input SAIS contexts
pub struct MultiThreaded8InputSaisContext {
    pub(crate) ptr: *mut c_void,
    pub(crate) num_threads: u16,
}

impl MultiThreaded8InputSaisContext {
    pub fn new(num_threads: u16) -> Self {
        // SAFETY: constructing the context is not not unsafe
        Self {
            ptr: unsafe { libsais::libsais_create_ctx_omp(num_threads.into()) },
            num_threads,
        }
    }
}

impl sealed::Sealed for MultiThreaded8InputSaisContext {}

impl SaisContext for MultiThreaded8InputSaisContext {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn num_threads(&self) -> u16 {
        self.num_threads
    }
}

impl Drop for MultiThreaded8InputSaisContext {
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        unsafe {
            libsais::libsais_free_ctx(self.ptr);
        }
    }
}

/// There is currently no use for the multithreaded 16-bit input SAIS contexts
pub struct MultiThreaded16InputSaisContext {
    pub(crate) ptr: *mut c_void,
    pub(crate) num_threads: u16,
}

impl MultiThreaded16InputSaisContext {
    pub fn new(num_threads: u16) -> Self {
        // SAFETY: constructing the context is not not unsafe
        Self {
            ptr: unsafe { libsais16::libsais16_create_ctx_omp(num_threads.into()) },
            num_threads,
        }
    }
}

impl sealed::Sealed for MultiThreaded16InputSaisContext {}

impl SaisContext for MultiThreaded16InputSaisContext {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn num_threads(&self) -> u16 {
        self.num_threads
    }
}

impl Drop for MultiThreaded16InputSaisContext {
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        unsafe {
            libsais16::libsais16_free_ctx(self.ptr);
        }
    }
}

pub struct ContextUnimplemented {}

impl sealed::Sealed for ContextUnimplemented {}

impl SaisContext for ContextUnimplemented {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        unimplemented!("Using a context is not implemented for this input type.");
    }

    fn num_threads(&self) -> u16 {
        unimplemented!("Using a context is not implemented for this input type.");
    }
}
