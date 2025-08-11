use std::ffi::c_void;

use libsais_sys::libsais;

use crate::context::{self, SaisContext};

pub trait Parallelism {
    type Context: SaisContext;

    unsafe fn run_libsais(
        text_ptr: *const u8,
        suffix_array_buffer_ptr: *mut i32,
        text_len: i32,
        extra_space: i32,
        frequency_table_ptr: *mut i32,
        num_threads: i32,
        generalized_suffix_array: bool,
        context: Option<*mut c_void>,
    ) -> i32;
}

pub(crate) enum SingleThreaded {}

impl Parallelism for SingleThreaded {
    type Context = context::SingleThreadedSaisContext;

    unsafe fn run_libsais(
        text_ptr: *const u8,
        suffix_array_buffer_ptr: *mut i32,
        text_len: i32,
        extra_space: i32,
        frequency_table_ptr: *mut i32,
        _num_threads: i32,
        generalized_suffix_array: bool,
        context: Option<*mut c_void>,
    ) -> i32 {
        unsafe {
            match (generalized_suffix_array, context) {
                (true, None) => libsais::libsais_gsa(
                    text_ptr,
                    suffix_array_buffer_ptr,
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                ),
                (true, Some(context)) => libsais::libsais_gsa_ctx(
                    context,
                    text_ptr,
                    suffix_array_buffer_ptr,
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                ),
                (false, None) => libsais::libsais(
                    text_ptr,
                    suffix_array_buffer_ptr,
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                ),
                (false, Some(context)) => libsais::libsais_ctx(
                    context,
                    text_ptr,
                    suffix_array_buffer_ptr,
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                ),
            }
        }
    }
}

#[cfg(feature = "openmp")]
pub(crate) enum MultiThreaded {}

#[cfg(feature = "openmp")]
impl Parallelism for MultiThreaded {
    type Context = context::MultiThreadedSaisContext;

    unsafe fn run_libsais(
        text_ptr: *const u8,
        suffix_array_buffer_ptr: *mut i32,
        text_len: i32,
        extra_space: i32,
        frequency_table_ptr: *mut i32,
        num_threads: i32,
        generalized_suffix_array: bool,
        _context: Option<*mut c_void>, // currently, no libsais_omp_ctx functions exist
    ) -> i32 {
        unsafe {
            if generalized_suffix_array {
                libsais::libsais_gsa_omp(
                    text_ptr,
                    suffix_array_buffer_ptr,
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                    num_threads,
                )
            } else {
                libsais::libsais_omp(
                    text_ptr,
                    suffix_array_buffer_ptr,
                    text_len,
                    extra_space,
                    frequency_table_ptr,
                    num_threads,
                )
            }
        }
    }
}

// trait InputTextValue {}

// impl InputTextValue for u8 {} // libsais, libsais64
// impl InputTextValue for i32 {} // libsais
// impl InputTextValue for u16 {} // libsais16, libsais16x64
// impl InputTextValue for i64 {} // libsais64

// trait OutputBufferValue {}

// impl OutputBufferValue for i32 {} // libsais, libsais16
// impl OutputBufferValue for i64 {} // libsais64, libsais16x64
