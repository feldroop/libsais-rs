use libsais_sys::{libsais, libsais16, libsais16x64, libsais64};

use std::ffi::c_void;

use super::{
    FunctionsUnimplemented, LibsaisFunctionsLargeAlphabet, LibsaisFunctionsSmallAlphabet,
    LibsaisLcpFunctions, SingleThreaded8InputOutputDispatcher,
    SingleThreaded16InputOutputDispatcher, SingleThreaded32InputOutputDispatcher,
    SingleThreaded64InputOutputDispatcher,
};
use crate::{
    InputElement, LargeAlphabet, OutputElement, SmallAlphabet, sealed::Sealed,
    typestate::OutputElementOrUndecided,
};

#[cfg(feature = "openmp")]
use super::{
    MultiThreaded8InputOutputDispatcher, MultiThreaded16InputOutputDispatcher,
    MultiThreaded32InputOutputDispatcher, MultiThreaded64InputOutputDispatcher,
};

// this module could be considered the "backend" of this library.
// It implements the traits that model libsais functions for all combinations
// of Parallelism, u8/u16/i32/i64 as input elements and i32/i64 as output elements.
// The resulting types are used to implement the InputElement and OutputElement traits.
// When a combination of types is not supported, an unimplemented panic is inserted.
// These combinations should be impossible to create, which is guarded by the "frontend" builder API.
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
    ($mod_name:ident, $libsais_fn:ident,; $num_threads:ident, feature = "openmp") => {
        $mod_name::$libsais_fn(
            $num_threads
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
            unsafe fn libsais_plcp(
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

            unsafe fn libsais_lcp(
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
        $libsais_create_ctx_fn:ident,
        $libsais_free_ctx_fn:ident,
        $libsais_unbwt_fn:ident,
        $libsais_unbwt_ctx_fn:ident,
        $libsais_unbwt_aux_fn:ident,
        $libsais_unbwt_aux_ctx_fn:ident,
        $libsais_unbwt_create_ctx_fn:ident,
        $libsais_unbwt_free_ctx_fn:ident,
        $($parallelism_tail:tt)+
    ) => {
        #[cfg($($parallelism_tail)+)]
        pub struct $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl Sealed for $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl LibsaisFunctionsSmallAlphabet<$input_type, $output_type> for $struct_name {
            unsafe fn libsais(
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

            unsafe fn libsais_bwt(
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

            unsafe fn libsais_bwt_aux(
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

            unsafe fn libsais_create_ctx(_num_threads: $output_type) -> *mut c_void {
                #[allow(unused_unsafe)]
                unsafe {
                    fn_or_unimplemented_with_or_without_threads!(
                        $libsais_mod,
                        $libsais_create_ctx_fn,;
                        _num_threads,
                        $($parallelism_tail)+
                    )
                }
            }

            unsafe fn libsais_free_ctx(_ctx: *mut c_void) {
                #[allow(unused_unsafe)]
                unsafe {
                    fn_or_unimplemented!(
                        $libsais_mod,
                        $libsais_free_ctx_fn,
                        _ctx
                    )
                }
            }

            unsafe fn libsais_unbwt(
                bwt_ptr: *const $input_type,
                text_buffer_ptr: *mut $input_type,
                suffix_array_buffer_ptr: *mut $output_type,
                bwt_len: $output_type,
                frequency_table_ptr: *const $output_type,
                primary_index: $output_type,
                _num_threads: $output_type,
                context: Option<*mut c_void>,
            ) -> $output_type {
                unsafe {
                    if let Some(_context) = context {
                        fn_or_unimplemented!(
                            $libsais_mod,
                            $libsais_unbwt_ctx_fn,
                            _context,
                            bwt_ptr,
                            text_buffer_ptr,
                            suffix_array_buffer_ptr,
                            bwt_len,
                            frequency_table_ptr,
                            primary_index
                        )
                    } else {
                        fn_with_or_without_threads!(
                            $libsais_mod,
                            $libsais_unbwt_fn,
                            bwt_ptr,
                            text_buffer_ptr,
                            suffix_array_buffer_ptr,
                            bwt_len,
                            frequency_table_ptr,
                            primary_index;
                            _num_threads,
                            $($parallelism_tail)+
                        )
                    }
                }
            }

            unsafe fn libsais_unbwt_aux(
                bwt_ptr: *const $input_type,
                text_buffer_ptr: *mut $input_type,
                suffix_array_buffer_ptr: *mut $output_type,
                bwt_len: $output_type,
                frequency_table_ptr: *const $output_type,
                aux_indices_sampling_rate: $output_type,
                aux_indices_buffer_ptr: *const $output_type,
                _num_threads: $output_type,
                context: Option<*mut c_void>,
            ) -> $output_type {
                unsafe {
                    if let Some(_context) = context {
                        fn_or_unimplemented!(
                            $libsais_mod,
                            $libsais_unbwt_aux_ctx_fn,
                            _context,
                            bwt_ptr,
                            text_buffer_ptr,
                            suffix_array_buffer_ptr,
                            bwt_len,
                            frequency_table_ptr,
                            aux_indices_sampling_rate,
                            aux_indices_buffer_ptr
                        )
                    } else {
                        fn_with_or_without_threads!(
                            $libsais_mod,
                            $libsais_unbwt_aux_fn,
                            bwt_ptr,
                            text_buffer_ptr,
                            suffix_array_buffer_ptr,
                            bwt_len,
                            frequency_table_ptr,
                            aux_indices_sampling_rate,
                            aux_indices_buffer_ptr;
                            _num_threads,
                            $($parallelism_tail)+
                        )
                    }
                }
            }

            unsafe fn libsais_unbwt_create_ctx(_num_threads: $output_type) -> *mut c_void {
                #[allow(unused_unsafe)]
                unsafe {
                    fn_or_unimplemented_with_or_without_threads!(
                        $libsais_mod,
                        $libsais_unbwt_create_ctx_fn,;
                        _num_threads,
                        $($parallelism_tail)+
                    )
                }
            }

            unsafe fn libsais_unbwt_free_ctx(_ctx: *mut c_void) {
                #[allow(unused_unsafe)]
                unsafe {
                    fn_or_unimplemented!(
                        $libsais_mod,
                        $libsais_unbwt_free_ctx_fn,
                        _ctx
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
    };
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
        impl Sealed for $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl LibsaisFunctionsLargeAlphabet<$input_type, $output_type> for $struct_name {
            unsafe fn libsais_large_alphabet(
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
    libsais_create_ctx,
    libsais_free_ctx,
    libsais_unbwt,
    libsais_unbwt_ctx,
    libsais_unbwt_aux,
    libsais_unbwt_aux_ctx,
    libsais_unbwt_create_ctx,
    libsais_unbwt_free_ctx,
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
    unimplemented,
    unimplemented,
    libsais64_unbwt,
    unimplemented,
    libsais64_unbwt_aux,
    unimplemented,
    unimplemented,
    unimplemented,
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
    libsais16_create_ctx,
    libsais16_free_ctx,
    libsais16_unbwt,
    libsais16_unbwt_ctx,
    libsais16_unbwt_aux,
    libsais16_unbwt_aux_ctx,
    libsais16_unbwt_create_ctx,
    libsais16_unbwt_free_ctx,
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
    unimplemented,
    unimplemented,
    libsais16x64_unbwt,
    unimplemented,
    libsais16x64_unbwt_aux,
    unimplemented,
    unimplemented,
    unimplemented,
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
    libsais_ctx,
    libsais_gsa_ctx,
    libsais_bwt_ctx,
    libsais_bwt_aux_ctx,
    libsais_plcp_omp,
    libsais_plcp_gsa_omp,
    libsais_lcp_omp,
    libsais_create_ctx_omp,
    libsais_free_ctx,
    libsais_unbwt_omp,
    libsais_unbwt_ctx,
    libsais_unbwt_aux_omp,
    libsais_unbwt_aux_ctx,
    libsais_unbwt_create_ctx_omp,
    libsais_unbwt_free_ctx,
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
    unimplemented,
    unimplemented,
    libsais64_unbwt_omp,
    unimplemented,
    libsais64_unbwt_aux_omp,
    unimplemented,
    unimplemented,
    unimplemented,
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
    libsais16_ctx,
    libsais16_gsa_ctx,
    libsais16_bwt_ctx,
    libsais16_bwt_aux_ctx,
    libsais16_plcp_omp,
    libsais16_plcp_gsa_omp,
    libsais16_lcp_omp,
    libsais16_create_ctx_omp,
    libsais16_free_ctx,
    libsais16_unbwt_omp,
    libsais16_unbwt_ctx,
    libsais16_unbwt_aux_omp,
    libsais16_unbwt_aux_ctx,
    libsais16_unbwt_create_ctx_omp,
    libsais16_unbwt_free_ctx,
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
    unimplemented,
    unimplemented,
    libsais16x64_unbwt_omp,
    unimplemented,
    libsais16x64_unbwt_aux_omp,
    unimplemented,
    unimplemented,
    unimplemented,
    feature = "openmp"
);

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

impl Sealed for u8 {}

impl InputElement for u8 {
    const RECOMMENDED_EXTRA_SPACE: usize = 0;

    type SingleThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        SingleThreaded8InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        MultiThreaded8InputOutputDispatcher<O>;
}

impl Sealed for u16 {}

impl InputElement for u16 {
    const RECOMMENDED_EXTRA_SPACE: usize = 0;

    type SingleThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        SingleThreaded16InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        MultiThreaded16InputOutputDispatcher<O>;
}

impl Sealed for i32 {}

impl InputElement for i32 {
    const RECOMMENDED_EXTRA_SPACE: usize = 6_000;

    type SingleThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        SingleThreaded32InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        MultiThreaded32InputOutputDispatcher<O>;
}

impl OutputElement for i32 {
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

impl Sealed for i64 {}

impl InputElement for i64 {
    const RECOMMENDED_EXTRA_SPACE: usize = 6_000;

    type SingleThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        SingleThreaded64InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementOrUndecided> =
        MultiThreaded64InputOutputDispatcher<O>;
}

impl OutputElement for i64 {
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

impl SmallAlphabet for u8 {
    const FREQUENCY_TABLE_SIZE: usize = 256;
}

impl SmallAlphabet for u16 {
    const FREQUENCY_TABLE_SIZE: usize = 65536;
}

impl LargeAlphabet for i32 {}
impl LargeAlphabet for i64 {}
