use libsais_sys::libsais;

use std::ffi::c_void;

mod sealed {
    pub trait Sealed {}
}
pub trait SaisContext: sealed::Sealed {
    fn as_mut_ptr(&mut self) -> *mut c_void;

    fn num_threads(&self) -> u16;
}

pub struct SingleThreadedSaisContext {
    pub(crate) ptr: *mut c_void,
}

impl SingleThreadedSaisContext {
    pub fn new() -> Self {
        // SAFETY: constructing the context is not not unsafe
        Self {
            ptr: unsafe { libsais::libsais_create_ctx() },
        }
    }
}

impl sealed::Sealed for SingleThreadedSaisContext {}

impl SaisContext for SingleThreadedSaisContext {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn num_threads(&self) -> u16 {
        1
    }
}

impl Drop for SingleThreadedSaisContext {
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        unsafe {
            libsais::libsais_free_ctx(self.ptr);
        }
    }
}

pub struct MultiThreadedSaisContext {
    pub(crate) ptr: *mut c_void,
    pub(crate) num_threads: u16,
}

impl MultiThreadedSaisContext {
    pub fn new(num_threads: u16) -> Self {
        // SAFETY: constructing the context is not not unsafe
        Self {
            ptr: unsafe { libsais::libsais_create_ctx_omp(num_threads.into()) },
            num_threads,
        }
    }
}

impl sealed::Sealed for MultiThreadedSaisContext {}

impl SaisContext for MultiThreadedSaisContext {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn num_threads(&self) -> u16 {
        self.num_threads
    }
}

impl Drop for MultiThreadedSaisContext {
    fn drop(&mut self) {
        // SAFETY: this pointer was acquired by calling one of the corresponding create_ctx functions
        unsafe {
            libsais::libsais_free_ctx(self.ptr);
        }
    }
}
