use std::{ffi::c_void, marker::PhantomData};

use libsais_sys::{libsais, libsais16, libsais16x64, libsais64};

use crate::context::{
    ContextUnimplemented, SaisContext, SingleThreaded8InputSaisContext,
    SingleThreaded16InputSaisContext,
};

pub trait LibsaisFunctionsSmallAlphabet<I: InputElement, O: OutputElementDecided>:
    sealed::Sealed
{
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

macro_rules! libsais_fn_or_unimplemented {
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
                            libsais_fn_or_unimplemented!(
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
                            libsais_fn_or_unimplemented!(
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
                        libsais_fn_or_unimplemented!(
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
                        libsais_fn_or_unimplemented!(
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
    feature = "openmp"
);

pub trait LibsaisFunctionsLargeAlphabet<I: InputElement, O: OutputElementDecided>:
    sealed::Sealed
{
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
    }
}

libsais_functions_large_alphabet_impl!(
    SingleThreaded32Input32Output,
    i32,
    i32,
    libsais,
    libsais_int,
    all()
);

libsais_functions_large_alphabet_impl!(
    SingleThreaded64Input64Output,
    i64,
    i64,
    libsais64,
    libsais64_long,
    all()
);

libsais_functions_large_alphabet_impl!(
    MultiThreaded32Input32Output,
    i32,
    i32,
    libsais,
    libsais_int_omp,
    feature = "openmp"
);

libsais_functions_large_alphabet_impl!(
    MultiThreaded64Input64Output,
    i64,
    i64,
    libsais64,
    libsais64_long_omp,
    feature = "openmp"
);

// -------------------- placeholder type for output dispatch --------------------
pub struct FunctionsUnimplemented {}

impl sealed::Sealed for FunctionsUnimplemented {}

impl<I: InputElement, O: OutputElementDecided> LibsaisFunctionsSmallAlphabet<I, O>
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

impl<I: InputElement, O: OutputElementDecided> LibsaisFunctionsLargeAlphabet<I, O>
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

// -------------------- Typestate traits for Builder API --------------------
pub enum Undecided {}

impl sealed::Sealed for Undecided {}

pub trait Parallelism: sealed::Sealed {
    type WithInput<I: InputElement, O: OutputElementDecided>: InputDispatch<I, O>;
}

pub enum SingleThreaded {}

impl sealed::Sealed for SingleThreaded {}

impl Parallelism for SingleThreaded {
    type WithInput<I: InputElement, O: OutputElementDecided> = SingleThreadedInputDispatcher<I, O>;
}

#[cfg(feature = "openmp")]
pub enum MultiThreaded {}

#[cfg(feature = "openmp")]
impl sealed::Sealed for MultiThreaded {}

#[cfg(feature = "openmp")]
impl Parallelism for MultiThreaded {
    type WithInput<I: InputElement, O: OutputElementDecided> = MultiThreadedInputDispatcher<I, O>;
}

pub trait OutputElement: sealed::Sealed {}

impl OutputElement for Undecided {}

pub trait BufferMode: sealed::Sealed {}

impl BufferMode for Undecided {}

pub struct BorrowedBuffer {}

impl sealed::Sealed for BorrowedBuffer {}

impl BufferMode for BorrowedBuffer {}

pub struct OwnedBuffer {}

impl sealed::Sealed for OwnedBuffer {}

impl BufferMode for OwnedBuffer {}

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

// -------------------- InputElementDecided and OutputElementDecided with implementations for u8, u16, i32, i64 --------------------
// Unsafe trait because the context type has to be correct for this input type
pub unsafe trait InputElement:
    sealed::Sealed + Copy + TryFrom<usize, Error: std::fmt::Debug> + Into<i64> + Clone + Ord
{
    const RECOMMENDED_EXTRA_SPACE: usize;

    type SingleThreadedContext: SaisContext;
    type SingleThreadedOutputDispatcher<O: OutputElementDecided>: OutputDispatch<Self, O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementDecided>: OutputDispatch<Self, O>;
}

pub trait OutputElementDecided:
    sealed::Sealed
    + Copy
    + TryFrom<usize, Error: std::fmt::Debug>
    + Into<i64>
    + Clone
    + std::fmt::Display
{
    const MAX: Self;

    type SingleThreaded8InputFunctions: LibsaisFunctionsSmallAlphabet<u8, Self>;
    type SingleThreaded16InputFunctions: LibsaisFunctionsSmallAlphabet<u16, Self>;
    type SingleThreaded32InputFunctions: LibsaisFunctionsLargeAlphabet<i32, Self>;
    type SingleThreaded64InputFunctions: LibsaisFunctionsLargeAlphabet<i64, Self>;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions: LibsaisFunctionsSmallAlphabet<u8, Self>;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions: LibsaisFunctionsSmallAlphabet<u16, Self>;
    #[cfg(feature = "openmp")]
    type MultiThreaded32InputFunctions: LibsaisFunctionsLargeAlphabet<i32, Self>;
    #[cfg(feature = "openmp")]
    type MultiThreaded64InputFunctions: LibsaisFunctionsLargeAlphabet<i64, Self>;
}

impl<B: OutputElementDecided> OutputElement for B {}

impl sealed::Sealed for u8 {}

unsafe impl InputElement for u8 {
    const RECOMMENDED_EXTRA_SPACE: usize = 0;

    type SingleThreadedContext = SingleThreaded8InputSaisContext;
    type SingleThreadedOutputDispatcher<O: OutputElementDecided> =
        SingleThreaded8InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementDecided> =
        MultiThreaded8InputOutputDispatcher<O>;
}

impl sealed::Sealed for u16 {}

unsafe impl InputElement for u16 {
    const RECOMMENDED_EXTRA_SPACE: usize = 0;

    type SingleThreadedContext = SingleThreaded16InputSaisContext;
    type SingleThreadedOutputDispatcher<O: OutputElementDecided> =
        SingleThreaded16InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementDecided> =
        MultiThreaded16InputOutputDispatcher<O>;
}

impl sealed::Sealed for i32 {}

unsafe impl InputElement for i32 {
    const RECOMMENDED_EXTRA_SPACE: usize = 6_000;

    type SingleThreadedContext = ContextUnimplemented;
    type SingleThreadedOutputDispatcher<O: OutputElementDecided> =
        SingleThreaded32InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementDecided> =
        MultiThreaded32InputOutputDispatcher<O>;
}

impl OutputElementDecided for i32 {
    const MAX: Self = Self::MAX;

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

    type SingleThreadedContext = ContextUnimplemented;
    type SingleThreadedOutputDispatcher<O: OutputElementDecided> =
        SingleThreaded64InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementDecided> =
        MultiThreaded64InputOutputDispatcher<O>;
}

impl OutputElementDecided for i64 {
    const MAX: Self = Self::MAX;

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

pub trait IsValidOutputFor<I: InputElement>: sealed::Sealed + OutputElementDecided {}

impl IsValidOutputFor<u8> for i32 {}
impl IsValidOutputFor<u16> for i32 {}
impl IsValidOutputFor<i32> for i32 {}

impl IsValidOutputFor<u8> for i64 {}
impl IsValidOutputFor<u16> for i64 {}
impl IsValidOutputFor<i64> for i64 {}

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

pub trait LargeAlphabet: InputElement + OutputElementDecided {}

impl LargeAlphabet for i32 {}
impl LargeAlphabet for i64 {}

// -------------------- InputDispatch and implementations --------------------
pub trait InputDispatch<I: InputElement, O: OutputElementDecided>: sealed::Sealed {
    type WithOutput: OutputDispatch<I, O>;
}

pub struct SingleThreadedInputDispatcher<I, O> {
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

impl<I, O> sealed::Sealed for SingleThreadedInputDispatcher<I, O> {}

impl<I: InputElement, O: OutputElementDecided> InputDispatch<I, O>
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
impl<I: InputElement, O: OutputElementDecided> InputDispatch<I, O>
    for MultiThreadedInputDispatcher<I, O>
{
    type WithOutput = I::MultiThreadedOutputDispatcher<O>;
}

// -------------------- OutputDispatch and implementations --------------------
pub trait OutputDispatch<I: InputElement, O: OutputElementDecided>: sealed::Sealed {
    type SmallAlphabetFunctions: LibsaisFunctionsSmallAlphabet<I, O>;
    type LargeAlphabetFunctions: LibsaisFunctionsLargeAlphabet<I, O>;
}

pub struct SingleThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded8InputOutputDispatcher<O> {}

impl<O: OutputElementDecided> OutputDispatch<u8, O> for SingleThreaded8InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::SingleThreaded8InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
}

pub struct SingleThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded16InputOutputDispatcher<O> {}

impl<O: OutputElementDecided> OutputDispatch<u16, O> for SingleThreaded16InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::SingleThreaded16InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
}

pub struct SingleThreaded32InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded32InputOutputDispatcher<O> {}

impl<O: OutputElementDecided> OutputDispatch<i32, O> for SingleThreaded32InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::SingleThreaded32InputFunctions;
}

pub struct SingleThreaded64InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> sealed::Sealed for SingleThreaded64InputOutputDispatcher<O> {}

impl<O: OutputElementDecided> OutputDispatch<i64, O> for SingleThreaded64InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::SingleThreaded64InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded8InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementDecided> OutputDispatch<u8, O> for MultiThreaded8InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::MultiThreaded8InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded16InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementDecided> OutputDispatch<u16, O> for MultiThreaded16InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::MultiThreaded16InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded32InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded32InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementDecided> OutputDispatch<i32, O> for MultiThreaded32InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::MultiThreaded32InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded64InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> sealed::Sealed for MultiThreaded64InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementDecided> OutputDispatch<i64, O> for MultiThreaded64InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::MultiThreaded64InputFunctions;
}

mod sealed {
    pub trait Sealed {}
}
