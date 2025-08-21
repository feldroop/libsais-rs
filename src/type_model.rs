use std::{ffi::c_void, marker::PhantomData};

use either::Either;
use libsais_sys::{libsais, libsais16, libsais16x64, libsais64};

use crate::context::{
    ContextUnimplemented, SaisContext, SingleThreaded8InputSaisContext,
    SingleThreaded16InputSaisContext,
};

pub trait LibsaisFunctionsSmallAlphabet<I: InputElement, O: OutputElement>: sealed::Sealed {
    unsafe fn run_libsais(
        text_ptr: *const I,
        suffix_array_buffer_ptr: *mut O,
        text_len: O,
        extra_space: O,
        frequency_table_ptr: *mut O,
        num_threads: O,
        generalized_suffix_array: bool,
        context: Option<*mut c_void>,
    ) -> O;

    unsafe fn run_libsais_bwt(
        text_ptr: *const I,
        bwt_buffer_ptr: *mut I,
        suffix_array_buffer_ptr: *mut O,
        text_len: O,
        extra_space: O,
        frequency_table_ptr: *mut O,
        num_threads: O,
        context: Option<*mut c_void>,
    ) -> O;

    unsafe fn run_libsais_bwt_aux(
        text_ptr: *const I,
        bwt_buffer_ptr: *mut I,
        suffix_array_buffer_ptr: *mut O,
        text_len: O,
        extra_space: O,
        frequency_table_ptr: *mut O,
        aux_indices_sampling_rate: O,
        aux_indices_buffer_ptr: *mut O,
        num_threads: O,
        context: Option<*mut c_void>,
    ) -> O;
}

pub trait LibsaisLcpFunctions<I: InputElement, O: OutputElement>: sealed::Sealed {
    unsafe fn run_libsais_plcp(
        text_ptr: *const I,
        suffix_array_ptr: *const O,
        plcp_ptr: *mut O,
        text_len: O,
        num_threads: O,
        generalized_suffix_array: bool,
    ) -> O;

    unsafe fn run_libsais_lcp(
        plcp_ptr: *const O,
        suffix_array_ptr: *const O,
        lcp_ptr: *mut O,
        suffix_array_len: O,
        num_threads: O,
    ) -> O;
}

macro_rules! fn_or_unimplemented {
    ($mod_name:ident, unimplemented, $($parameter:ident),*) => {
        unimplemented!("This function is currently not implemented by {}.", stringify!($mod_name))
    };
    ($mod_name:ident, $libsais_fn:ident, $($parameter:ident),*) => {
        $mod_name::$libsais_fn(
            $($parameter),*
        )
    };
}

macro_rules! fn_with_or_without_threads {
    ($mod_name:ident, $libsais_fn:ident, $($parameter:ident),*; $num_threads:ident, all()) => {
        $mod_name::$libsais_fn(
            $($parameter),*
        )
    };
    ($mod_name:ident, $libsais_fn:ident, $($parameter:ident),*; $num_threads:ident, feature = "openmp") => {
        $mod_name::$libsais_fn(
            $($parameter),*,
            $num_threads
        )
    };
}

macro_rules! fn_or_unimplemented_with_or_without_threads {
    ($mod_name:ident, unimplemented, $($tail:tt)+) => {
        unimplemented!("This function is currently not implemented by {}.", stringify!($mod_name))
    };
    ($mod_name:ident, $libsais_fn:ident, $($tail:tt)+) => {
        fn_with_or_without_threads!($mod_name, $libsais_fn, $($tail)+)
    };
}

macro_rules! lcp_functions_impl {
    (
        $struct_name:ident,
        $input_type:ty,
        $output_type:ty,
        $libsais_mod:ident,
        $libsais_plcp_fn:ident,
        $libsais_plcp_gsa_fn:ident,
        $libsais_lcp_fn:ident,
        $($parallelism_tail:tt)+
    ) => {
        #[cfg($($parallelism_tail)+)]
        impl LibsaisLcpFunctions<$input_type, $output_type> for $struct_name {
            unsafe fn run_libsais_plcp(
                _text_ptr: *const $input_type,
                _suffix_array_ptr: *const $output_type,
                _plcp_ptr: *mut $output_type,
                _text_len: $output_type,
                _num_threads: $output_type,
                generalized_suffix_array: bool,
            ) -> $output_type {
                #[allow(unused_unsafe)]
                unsafe {
                    if generalized_suffix_array {
                        fn_or_unimplemented_with_or_without_threads!(
                            $libsais_mod,
                            $libsais_plcp_gsa_fn,
                            _text_ptr,
                            _suffix_array_ptr,
                            _plcp_ptr,
                            _text_len;
                            _num_threads,
                            $($parallelism_tail)+
                        )
                    } else {
                        fn_or_unimplemented_with_or_without_threads!(
                            $libsais_mod,
                            $libsais_plcp_fn,
                            _text_ptr,
                            _suffix_array_ptr,
                            _plcp_ptr,
                            _text_len;
                            _num_threads,
                            $($parallelism_tail)+
                        )
                    }
                }
            }

            unsafe fn run_libsais_lcp(
                plcp_ptr: *const $output_type,
                suffix_array_ptr: *const $output_type,
                lcp_ptr: *mut $output_type,
                suffix_array_len: $output_type,
                _num_threads: $output_type,
            ) -> $output_type {
                unsafe {
                    fn_with_or_without_threads!(
                        $libsais_mod,
                        $libsais_lcp_fn,
                        plcp_ptr,
                        suffix_array_ptr,
                        lcp_ptr,
                        suffix_array_len;
                        _num_threads,
                        $($parallelism_tail)+
                    )
                }
            }
        }
    };
}

macro_rules! libsais_functions_small_alphabet_impl {
    (
        $struct_name:ident,
        $input_type:ty,
        $output_type:ty,
        $libsais_mod:ident,
        $libsais_fn:ident,
        $libsais_gsa_fn:ident,
        $libsais_bwt_fn:ident,
        $libsais_bwt_aux_fn:ident,
        $libsais_ctx_fn:ident,
        $libsais_gsa_ctx_fn:ident,
        $libsais_bwt_ctx_fn:ident,
        $libsais_bwt_aux_ctx_fn:ident,
        $libsais_plcp_fn:ident,
        $libsais_plcp_gsa_fn:ident,
        $libsais_lcp_fn:ident,
        $($parallelism_tail:tt)+
    ) => {
        #[cfg($($parallelism_tail)+)]
        pub struct $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl sealed::Sealed for $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl LibsaisFunctionsSmallAlphabet<$input_type, $output_type> for $struct_name {
            unsafe fn run_libsais(
                text_ptr: *const $input_type,
                suffix_array_buffer_ptr: *mut $output_type,
                text_len: $output_type,
                extra_space: $output_type,
                frequency_table_ptr: *mut $output_type,
                _num_threads: $output_type,
                generalized_suffix_array: bool,
                context: Option<*mut c_void>,
            ) -> $output_type {
                unsafe {
                    match (generalized_suffix_array, context) {
                        (true, None) => fn_with_or_without_threads!(
                            $libsais_mod,
                            $libsais_gsa_fn,
                            text_ptr,
                            suffix_array_buffer_ptr,
                            text_len,
                            extra_space,
                            frequency_table_ptr;
                            _num_threads,
                            $($parallelism_tail)+
                        ),
                        (true, Some(_context)) => {
                            fn_or_unimplemented!(
                                $libsais_mod,
                                $libsais_gsa_ctx_fn,
                                _context,
                                text_ptr,
                                suffix_array_buffer_ptr,
                                text_len,
                                extra_space,
                                frequency_table_ptr
                            )
                        }
                        (false, None) =>
                            fn_with_or_without_threads!(
                                $libsais_mod,
                                $libsais_fn,
                                text_ptr,
                                suffix_array_buffer_ptr,
                                text_len,
                                extra_space,
                                frequency_table_ptr;
                                _num_threads,
                                $($parallelism_tail)+
                            ),
                        (false, Some(_context)) => {
                            fn_or_unimplemented!(
                                $libsais_mod,
                                $libsais_ctx_fn,
                                _context,
                                text_ptr,
                                suffix_array_buffer_ptr,
                                text_len,
                                extra_space,
                                frequency_table_ptr
                            )
                        }
                    }
                }
            }

            unsafe fn run_libsais_bwt(
                text_ptr: *const $input_type,
                bwt_buffer_ptr: *mut $input_type,
                suffix_array_buffer_ptr: *mut $output_type,
                text_len: $output_type,
                extra_space: $output_type,
                frequency_table_ptr: *mut $output_type,
                _num_threads: $output_type,
                context: Option<*mut c_void>,
            ) -> $output_type {
                unsafe {
                    if let Some(_context) = context {
                        fn_or_unimplemented!(
                            $libsais_mod,
                            $libsais_bwt_ctx_fn,
                            _context,
                            text_ptr,
                            bwt_buffer_ptr,
                            suffix_array_buffer_ptr,
                            text_len,
                            extra_space,
                            frequency_table_ptr
                        )
                    } else {
                        fn_with_or_without_threads!(
                            $libsais_mod,
                            $libsais_bwt_fn,
                            text_ptr,
                            bwt_buffer_ptr,
                            suffix_array_buffer_ptr,
                            text_len,
                            extra_space,
                            frequency_table_ptr;
                            _num_threads,
                            $($parallelism_tail)+
                        )
                    }
                }
            }

            unsafe fn run_libsais_bwt_aux(
                text_ptr: *const $input_type,
                bwt_buffer_ptr: *mut $input_type,
                suffix_array_buffer_ptr: *mut $output_type,
                text_len: $output_type,
                extra_space: $output_type,
                frequency_table_ptr: *mut $output_type,
                aux_indices_sampling_rate: $output_type,
                aux_indices_buffer_ptr: *mut $output_type,
                _num_threads: $output_type,
                context: Option<*mut c_void>,
            ) -> $output_type{
                unsafe {
                    if let Some(_context) = context {
                        fn_or_unimplemented!(
                            $libsais_mod,
                            $libsais_bwt_aux_ctx_fn,
                            _context,
                            text_ptr,
                            bwt_buffer_ptr,
                            suffix_array_buffer_ptr,
                            text_len,
                            extra_space,
                            frequency_table_ptr,
                            aux_indices_sampling_rate,
                            aux_indices_buffer_ptr
                        )
                    } else {
                        fn_with_or_without_threads!(
                            $libsais_mod,
                            $libsais_bwt_aux_fn,
                            text_ptr,
                            bwt_buffer_ptr,
                            suffix_array_buffer_ptr,
                            text_len,
                            extra_space,
                            frequency_table_ptr,
                            aux_indices_sampling_rate,
                            aux_indices_buffer_ptr;
                            _num_threads,
                            $($parallelism_tail)+
                        )
                    }
                }
            }
        }

        lcp_functions_impl!(
            $struct_name,
            $input_type,
            $output_type,
            $libsais_mod,
            $libsais_plcp_fn,
            $libsais_plcp_gsa_fn,
            $libsais_lcp_fn,
            $($parallelism_tail)+
        );
    };
}

libsais_functions_small_alphabet_impl!(
    SingleThreaded8Input32Output,
    u8,
    i32,
    libsais,
    libsais,
    libsais_gsa,
    libsais_bwt,
    libsais_bwt_aux,
    libsais_ctx,
    libsais_gsa_ctx,
    libsais_bwt_ctx,
    libsais_bwt_aux_ctx,
    libsais_plcp,
    libsais_plcp_gsa,
    libsais_lcp,
    all()
);

libsais_functions_small_alphabet_impl!(
    SingleThreaded8Input64Output,
    u8,
    i64,
    libsais64,
    libsais64,
    libsais64_gsa,
    libsais64_bwt,
    libsais64_bwt_aux,
    unimplemented,
    unimplemented,
    unimplemented,
    unimplemented,
    libsais64_plcp,
    libsais64_plcp_gsa,
    libsais64_lcp,
    all()
);

libsais_functions_small_alphabet_impl!(
    SingleThreaded16Input32Output,
    u16,
    i32,
    libsais16,
    libsais16,
    libsais16_gsa,
    libsais16_bwt,
    libsais16_bwt_aux,
    libsais16_ctx,
    libsais16_gsa_ctx,
    libsais16_bwt_ctx,
    libsais16_bwt_aux_ctx,
    libsais16_plcp,
    libsais16_plcp_gsa,
    libsais16_lcp,
    all()
);

libsais_functions_small_alphabet_impl!(
    SingleThreaded16Input64Output,
    u16,
    i64,
    libsais16x64,
    libsais16x64,
    libsais16x64_gsa,
    libsais16x64_bwt,
    libsais16x64_bwt_aux,
    unimplemented,
    unimplemented,
    unimplemented,
    unimplemented,
    libsais16x64_plcp,
    libsais16x64_plcp_gsa,
    libsais16x64_lcp,
    all()
);

libsais_functions_small_alphabet_impl!(
    MultiThreaded8Input32Output,
    u8,
    i32,
    libsais,
    libsais_omp,
    libsais_gsa_omp,
    libsais_bwt_omp,
    libsais_bwt_aux_omp,
    unimplemented,
    unimplemented,
    unimplemented,
    unimplemented,
    libsais_plcp_omp,
    libsais_plcp_gsa_omp,
    libsais_lcp_omp,
    feature = "openmp"
);

libsais_functions_small_alphabet_impl!(
    MultiThreaded8Input64Output,
    u8,
    i64,
    libsais64,
    libsais64_omp,
    libsais64_gsa_omp,
    libsais64_bwt_omp,
    libsais64_bwt_aux_omp,
    unimplemented,
    unimplemented,
    unimplemented,
    unimplemented,
    libsais64_plcp_omp,
    libsais64_plcp_gsa_omp,
    libsais64_lcp_omp,
    feature = "openmp"
);

libsais_functions_small_alphabet_impl!(
    MultiThreaded16Input32Output,
    u16,
    i32,
    libsais16,
    libsais16_omp,
    libsais16_gsa_omp,
    libsais16_bwt_omp,
    libsais16_bwt_aux_omp,
    unimplemented,
    unimplemented,
    unimplemented,
    unimplemented,
    libsais16_plcp_omp,
    libsais16_plcp_gsa_omp,
    libsais16_lcp_omp,
    feature = "openmp"
);

libsais_functions_small_alphabet_impl!(
    MultiThreaded16Input64Output,
    u16,
    i64,
    libsais16x64,
    libsais16x64_omp,
    libsais16x64_gsa_omp,
    libsais16x64_bwt_omp,
    libsais16x64_bwt_aux_omp,
    unimplemented,
    unimplemented,
    unimplemented,
    unimplemented,
    libsais16x64_plcp_omp,
    libsais16x64_plcp_gsa_omp,
    libsais16x64_lcp_omp,
    feature = "openmp"
);

pub trait LibsaisFunctionsLargeAlphabet<I: InputElement, O: OutputElement>: sealed::Sealed {
    unsafe fn run_libsais_large_alphabet(
        text_ptr: *mut I,
        suffix_array_buffer_ptr: *mut O,
        text_len: O,
        alphabet_size: O,
        extra_space: O,
        num_threads: O,
    ) -> O;
}

macro_rules! libsais_functions_large_alphabet_impl {
    (
        $struct_name:ident,
        $input_type:ty,
        $output_type:ty,
        $libsais_mod:ident,
        $libsais_fn:ident,
        $libsais_plcp_fn:ident,
        $libsais_plcp_gsa_fn:ident,
        $libsais_lcp_fn:ident,
        $($parallelism_tail:tt)+
    ) => {
        #[cfg($($parallelism_tail)+)]
        pub struct $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl sealed::Sealed for $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl LibsaisFunctionsLargeAlphabet<$input_type, $output_type> for $struct_name {
            unsafe fn run_libsais_large_alphabet(
                text_ptr: *mut $input_type,
                suffix_array_buffer_ptr: *mut $output_type,
                text_len: $output_type,
                alphabet_size: $output_type,
                extra_space: $output_type,
                _num_threads: $output_type,
            ) -> $output_type {
                unsafe {
                    fn_with_or_without_threads!(
                        $libsais_mod,
                        $libsais_fn,
                        text_ptr,
                        suffix_array_buffer_ptr,
                        text_len,
                        alphabet_size,
                        extra_space;
                        _num_threads,
                        $($parallelism_tail)+
                    )
                }
            }
        }

        lcp_functions_impl!(
            $struct_name,
            $input_type,
            $output_type,
            $libsais_mod,
            $libsais_plcp_fn,
            $libsais_plcp_gsa_fn,
            $libsais_lcp_fn,
            $($parallelism_tail)+
        );
    }
}

libsais_functions_large_alphabet_impl!(
    SingleThreaded32Input32Output,
    i32,
    i32,
    libsais,
    libsais_int,
    libsais_plcp_int,
    unimplemented,
    libsais_lcp,
    all()
);

libsais_functions_large_alphabet_impl!(
    SingleThreaded64Input64Output,
    i64,
    i64,
    libsais64,
    libsais64_long,
    unimplemented,
    unimplemented,
    libsais64_lcp,
    all()
);

libsais_functions_large_alphabet_impl!(
    MultiThreaded32Input32Output,
    i32,
    i32,
    libsais,
    libsais_int_omp,
    libsais_plcp_int_omp,
    unimplemented,
    libsais_lcp_omp,
    feature = "openmp"
);

libsais_functions_large_alphabet_impl!(
    MultiThreaded64Input64Output,
    i64,
    i64,
    libsais64,
    libsais64_long_omp,
    unimplemented,
    unimplemented,
    libsais64_lcp_omp,
    feature = "openmp"
);

// -------------------- placeholder type for output dispatch --------------------
pub struct FunctionsUnimplemented {}

impl sealed::Sealed for FunctionsUnimplemented {}

impl<I: InputElement, O: OutputElement> LibsaisFunctionsSmallAlphabet<I, O>
    for FunctionsUnimplemented
{
    unsafe fn run_libsais(
        _text_ptr: *const I,
        _suffix_array_buffer_ptr: *mut O,
        _text_len: O,
        _extra_space: O,
        _frequency_table_ptr: *mut O,
        _num_threads: O,
        _generalized_suffix_array: bool,
        _context: Option<*mut c_void>,
    ) -> O {
        unimplemented!(
            "There is no libsais implementation for this combination of input and output types.",
        );
    }

    unsafe fn run_libsais_bwt(
        _text_ptr: *const I,
        _bwt_buffer_ptr: *mut I,
        _suffix_array_buffer_ptr: *mut O,
        _text_len: O,
        _extra_space: O,
        _frequency_table_ptr: *mut O,
        _num_threads: O,
        _context: Option<*mut c_void>,
    ) -> O {
        unimplemented!(
            "There is no libsais bwt implementation for this combination of input and output types.",
        );
    }

    unsafe fn run_libsais_bwt_aux(
        _text_ptr: *const I,
        _bwt_buffer_ptr: *mut I,
        _suffix_array_buffer_ptr: *mut O,
        _text_len: O,
        _extra_space: O,
        _frequency_table_ptr: *mut O,
        _aux_indices_sampling_rate: O,
        _aux_indices_buffer_ptr: *mut O,
        _num_threads: O,
        _context: Option<*mut c_void>,
    ) -> O {
        unimplemented!(
            "There is no libsais bwt aux implementation for this combination of input and output types.",
        );
    }
}

impl<I: InputElement, O: OutputElement> LibsaisFunctionsLargeAlphabet<I, O>
    for FunctionsUnimplemented
{
    unsafe fn run_libsais_large_alphabet(
        _text_ptr: *mut I,
        _suffix_array_buffer_ptr: *mut O,
        _text_len: O,
        _alphabet_size: O,
        _extra_space: O,
        _num_threads: O,
    ) -> O {
        unimplemented!(
            "There is no libsais implementation for this combination of input and output types.",
        );
    }
}

impl<I: InputElement, O: OutputElement> LibsaisLcpFunctions<I, O> for FunctionsUnimplemented {
    unsafe fn run_libsais_plcp(
        _text_ptr: *const I,
        _suffix_array_ptr: *const O,
        _plcp_ptr: *mut O,
        _text_len: O,
        _num_threads: O,
        _generalized_suffix_array: bool,
    ) -> O {
        unimplemented!(
            "There is no libsais lcp implementation for this combination of input and output types."
        )
    }

    unsafe fn run_libsais_lcp(
        _plcp_ptr: *const O,
        _suffix_array_ptr: *const O,
        _lcp_ptr: *mut O,
        _text_len: O,
        _num_threads: O,
    ) -> O {
        unimplemented!(
            "There is no libsais lcp implementation for this combination of input and output types."
        )
    }
}

// -------------------- Typestate traits for Builder API --------------------
pub enum Undecided {}

impl sealed::Sealed for Undecided {}

pub trait Parallelism: sealed::Sealed {
    type WithInput<I: InputElement, O: OutputElement>: InputDispatch<I, O>;
}

pub enum SingleThreaded {}

impl sealed::Sealed for SingleThreaded {}

impl Parallelism for SingleThreaded {
    type WithInput<I: InputElement, O: OutputElement> = SingleThreadedInputDispatcher<I, O>;
}

#[cfg(feature = "openmp")]
pub enum MultiThreaded {}

#[cfg(feature = "openmp")]
impl sealed::Sealed for MultiThreaded {}

#[cfg(feature = "openmp")]
impl Parallelism for MultiThreaded {
    type WithInput<I: InputElement, O: OutputElement> = MultiThreadedInputDispatcher<I, O>;
}

pub trait OutputElementOrUndecided: sealed::Sealed {}

impl OutputElementOrUndecided for Undecided {}

pub trait BufferModeOrUndecided: sealed::Sealed {}

impl BufferModeOrUndecided for Undecided {}

pub trait BufferMode: sealed::Sealed {
    type Buffer<'a, T: 'a>;

    fn buffer_to_either<'a, T>(buffer: Self::Buffer<'a, T>) -> Either<Vec<T>, &'a mut [T]>;

    fn either_to_buffer<'a, T: std::fmt::Debug>(
        either_: Either<Vec<T>, &'a mut [T]>,
    ) -> Self::Buffer<'a, T>;

    fn unwrap_or_allocate<'a, T, F>(opt: Option<&'a mut [T]>, f: F) -> Self::Buffer<'a, T>
    where
        T: TryFrom<usize, Error: std::fmt::Debug> + Clone,
        F: FnOnce() -> Vec<T>;
}

impl<B: BufferMode> BufferModeOrUndecided for B {}

pub struct BorrowedBuffer {}

impl sealed::Sealed for BorrowedBuffer {}

impl BufferMode for BorrowedBuffer {
    type Buffer<'a, T: 'a> = &'a mut [T];

    fn buffer_to_either<'a, T>(buffer: Self::Buffer<'a, T>) -> Either<Vec<T>, &'a mut [T]> {
        Either::Right(buffer)
    }

    fn either_to_buffer<'a, T: std::fmt::Debug>(
        either_: Either<Vec<T>, &'a mut [T]>,
    ) -> Self::Buffer<'a, T> {
        either_.unwrap_right()
    }

    fn unwrap_or_allocate<'a, T, F>(opt: Option<&'a mut [T]>, _f: F) -> Self::Buffer<'a, T>
    where
        T: TryFrom<usize, Error: std::fmt::Debug> + Clone,
        F: FnOnce() -> Vec<T>,
    {
        opt.unwrap()
    }
}

pub struct OwnedBuffer {}

impl sealed::Sealed for OwnedBuffer {}

impl BufferMode for OwnedBuffer {
    type Buffer<'a, T: 'a> = Vec<T>;

    fn buffer_to_either<'a, T>(buffer: Self::Buffer<'a, T>) -> Either<Vec<T>, &'a mut [T]> {
        Either::Left(buffer)
    }

    fn either_to_buffer<'a, T: std::fmt::Debug>(
        either_: Either<Vec<T>, &'a mut [T]>,
    ) -> Self::Buffer<'a, T> {
        either_.unwrap_left()
    }

    fn unwrap_or_allocate<'a, T, F>(_opt: Option<&'a mut [T]>, f: F) -> Self::Buffer<'a, T>
    where
        T: TryFrom<usize, Error: std::fmt::Debug> + Clone,
        F: FnOnce() -> Vec<T>,
    {
        f()
    }
}

pub trait BufferModeOrReplaceInput: sealed::Sealed {}

impl<B: BufferMode> BufferModeOrReplaceInput for B {}

pub struct ReplaceInput {}

impl sealed::Sealed for ReplaceInput {}

impl BufferModeOrReplaceInput for ReplaceInput {}

pub trait AuxIndicesMode: sealed::Sealed {}

pub struct NoAuxIndices {}

impl sealed::Sealed for NoAuxIndices {}

impl AuxIndicesMode for NoAuxIndices {}

pub struct AuxIndicesBorrowedBuffer {}

impl sealed::Sealed for AuxIndicesBorrowedBuffer {}

impl AuxIndicesMode for AuxIndicesBorrowedBuffer {}

pub struct AuxIndicesOwnedBuffer {}

impl sealed::Sealed for AuxIndicesOwnedBuffer {}

impl AuxIndicesMode for AuxIndicesOwnedBuffer {}

// -------------------- InputElement and OutputElement with implementations for u8, u16, i32, i64 --------------------
// Unsafe trait because the context type has to be correct for this input type
pub unsafe trait InputElement:
    sealed::Sealed
    + std::fmt::Debug
    + Copy
    + TryFrom<usize, Error: std::fmt::Debug>
    + Into<i64>
    + Clone
    + Ord
{
    const RECOMMENDED_EXTRA_SPACE: usize;
    const ZERO: Self;

    type SingleThreadedContext: SaisContext;
    type SingleThreadedOutputDispatcher<O: OutputElement>: OutputDispatch<Self, O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElement>: OutputDispatch<Self, O>;
}

pub trait OutputElement:
    sealed::Sealed
    + std::fmt::Debug
    + Copy
    + TryFrom<usize, Error: std::fmt::Debug>
    + Into<i64>
    + Clone
    + std::fmt::Display
{
    const MAX: Self;
    const ZERO: Self;

    type SingleThreaded8InputFunctions: LibsaisFunctionsSmallAlphabet<u8, Self>
        + LibsaisLcpFunctions<u8, Self>;
    type SingleThreaded16InputFunctions: LibsaisFunctionsSmallAlphabet<u16, Self>
        + LibsaisLcpFunctions<u16, Self>;
    type SingleThreaded32InputFunctions: LibsaisFunctionsLargeAlphabet<i32, Self>
        + LibsaisLcpFunctions<i32, Self>;
    type SingleThreaded64InputFunctions: LibsaisFunctionsLargeAlphabet<i64, Self>
        + LibsaisLcpFunctions<i64, Self>;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions: LibsaisFunctionsSmallAlphabet<u8, Self>
        + LibsaisLcpFunctions<u8, Self>;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions: LibsaisFunctionsSmallAlphabet<u16, Self>
        + LibsaisLcpFunctions<u16, Self>;
    #[cfg(feature = "openmp")]
    type MultiThreaded32InputFunctions: LibsaisFunctionsLargeAlphabet<i32, Self>
        + LibsaisLcpFunctions<i32, Self>;
    #[cfg(feature = "openmp")]
    type MultiThreaded64InputFunctions: LibsaisFunctionsLargeAlphabet<i64, Self>
        + LibsaisLcpFunctions<i64, Self>;
}

impl<B: OutputElement> OutputElementOrUndecided for B {}

impl sealed::Sealed for u8 {}

unsafe impl InputElement for u8 {
    const RECOMMENDED_EXTRA_SPACE: usize = 0;
    const ZERO: Self = 0;

    type SingleThreadedContext = SingleThreaded8InputSaisContext;
    type SingleThreadedOutputDispatcher<O: OutputElement> = SingleThreaded8InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElement> = MultiThreaded8InputOutputDispatcher<O>;
}

impl sealed::Sealed for u16 {}

unsafe impl InputElement for u16 {
    const RECOMMENDED_EXTRA_SPACE: usize = 0;
    const ZERO: Self = 0;

    type SingleThreadedContext = SingleThreaded16InputSaisContext;
    type SingleThreadedOutputDispatcher<O: OutputElement> =
        SingleThreaded16InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElement> = MultiThreaded16InputOutputDispatcher<O>;
}

impl sealed::Sealed for i32 {}

unsafe impl InputElement for i32 {
    const RECOMMENDED_EXTRA_SPACE: usize = 6_000;
    const ZERO: Self = 0;

    type SingleThreadedContext = ContextUnimplemented;
    type SingleThreadedOutputDispatcher<O: OutputElement> =
        SingleThreaded32InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElement> = MultiThreaded32InputOutputDispatcher<O>;
}

impl OutputElement for i32 {
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;

    type SingleThreaded8InputFunctions = SingleThreaded8Input32Output;
    type SingleThreaded16InputFunctions = SingleThreaded16Input32Output;
    type SingleThreaded32InputFunctions = SingleThreaded32Input32Output;
    type SingleThreaded64InputFunctions = FunctionsUnimplemented;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions = MultiThreaded8Input32Output;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions = MultiThreaded16Input32Output;
    #[cfg(feature = "openmp")]
    type MultiThreaded32InputFunctions = MultiThreaded32Input32Output;
    #[cfg(feature = "openmp")]
    type MultiThreaded64InputFunctions = FunctionsUnimplemented;
}

impl sealed::Sealed for i64 {}

unsafe impl InputElement for i64 {
    const RECOMMENDED_EXTRA_SPACE: usize = 6_000;
    const ZERO: Self = 0;

    type SingleThreadedContext = ContextUnimplemented;
    type SingleThreadedOutputDispatcher<O: OutputElement> =
        SingleThreaded64InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElement> = MultiThreaded64InputOutputDispatcher<O>;
}

impl OutputElement for i64 {
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;

    type SingleThreaded8InputFunctions = SingleThreaded8Input64Output;
    type SingleThreaded16InputFunctions = SingleThreaded16Input64Output;
    type SingleThreaded32InputFunctions = FunctionsUnimplemented;
    type SingleThreaded64InputFunctions = SingleThreaded64Input64Output;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions = MultiThreaded8Input64Output;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions = MultiThreaded16Input64Output;
    #[cfg(feature = "openmp")]
    type MultiThreaded32InputFunctions = FunctionsUnimplemented;
    #[cfg(feature = "openmp")]
    type MultiThreaded64InputFunctions = MultiThreaded64Input64Output;
}

pub trait IsValidOutputFor<I: InputElement>: sealed::Sealed + OutputElement {}

impl IsValidOutputFor<u8> for i32 {}
impl IsValidOutputFor<u16> for i32 {}
impl IsValidOutputFor<i32> for i32 {}

impl IsValidOutputFor<u8> for i64 {}
impl IsValidOutputFor<u16> for i64 {}
impl IsValidOutputFor<i64> for i64 {}

pub trait SupportsPlcpOutputFor<I: InputElement>: sealed::Sealed + OutputElement {}

impl SupportsPlcpOutputFor<u8> for i32 {}
impl SupportsPlcpOutputFor<u16> for i32 {}
impl SupportsPlcpOutputFor<i32> for i32 {}

impl SupportsPlcpOutputFor<u8> for i64 {}
impl SupportsPlcpOutputFor<u16> for i64 {}

// -------------------- InputElementDecided refinement traits and implementations --------------------
pub trait SmallAlphabet: InputElement {
    const FREQUENCY_TABLE_SIZE: usize;
}

impl SmallAlphabet for u8 {
    const FREQUENCY_TABLE_SIZE: usize = 256;
}

impl SmallAlphabet for u16 {
    const FREQUENCY_TABLE_SIZE: usize = 65536;
}

pub trait LargeAlphabet: InputElement + OutputElement {}

impl LargeAlphabet for i32 {}
impl LargeAlphabet for i64 {}

// -------------------- InputDispatch and implementations --------------------
pub trait InputDispatch<I: InputElement, O: OutputElement>: sealed::Sealed {
    type WithOutput: OutputDispatch<I, O>;
}

pub struct SingleThreadedInputDispatcher<I, O> {
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

impl<I, O> sealed::Sealed for SingleThreadedInputDispatcher<I, O> {}

impl<I: InputElement, O: OutputElement> InputDispatch<I, O>
    for SingleThreadedInputDispatcher<I, O>
{
    type WithOutput = I::SingleThreadedOutputDispatcher<O>;
}

#[cfg(feature = "openmp")]
pub struct MultiThreadedInputDispatcher<I, O> {
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<I, O> sealed::Sealed for MultiThreadedInputDispatcher<I, O> {}

#[cfg(feature = "openmp")]
impl<I: InputElement, O: OutputElement> InputDispatch<I, O> for MultiThreadedInputDispatcher<I, O> {
    type WithOutput = I::MultiThreadedOutputDispatcher<O>;
}

// -------------------- OutputDispatch and implementations --------------------
pub trait OutputDispatch<I: InputElement, O: OutputElement>: sealed::Sealed {
    type SmallAlphabetFunctions: LibsaisFunctionsSmallAlphabet<I, O>;
    type LargeAlphabetFunctions: LibsaisFunctionsLargeAlphabet<I, O>;
    type LcpFunctions: LibsaisLcpFunctions<I, O>;
}

pub struct SingleThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded8InputOutputDispatcher<O> {}

impl<O: OutputElement> OutputDispatch<u8, O> for SingleThreaded8InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::SingleThreaded8InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::SingleThreaded8InputFunctions;
}

pub struct SingleThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded16InputOutputDispatcher<O> {}

impl<O: OutputElement> OutputDispatch<u16, O> for SingleThreaded16InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::SingleThreaded16InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::SingleThreaded16InputFunctions;
}

pub struct SingleThreaded32InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded32InputOutputDispatcher<O> {}

impl<O: OutputElement> OutputDispatch<i32, O> for SingleThreaded32InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::SingleThreaded32InputFunctions;
    type LcpFunctions = O::SingleThreaded32InputFunctions;
}

pub struct SingleThreaded64InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded64InputOutputDispatcher<O> {}

impl<O: OutputElement> OutputDispatch<i64, O> for SingleThreaded64InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::SingleThreaded64InputFunctions;
    type LcpFunctions = O::SingleThreaded64InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded8InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElement> OutputDispatch<u8, O> for MultiThreaded8InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::MultiThreaded8InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::MultiThreaded8InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded16InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElement> OutputDispatch<u16, O> for MultiThreaded16InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::MultiThreaded16InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::MultiThreaded16InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded32InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded32InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElement> OutputDispatch<i32, O> for MultiThreaded32InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::MultiThreaded32InputFunctions;
    type LcpFunctions = O::MultiThreaded32InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded64InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded64InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElement> OutputDispatch<i64, O> for MultiThreaded64InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::MultiThreaded64InputFunctions;
    type LcpFunctions = O::MultiThreaded64InputFunctions;
}

mod sealed {
    pub trait Sealed {}
}
