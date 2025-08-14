use std::{ffi::c_void, marker::PhantomData};

use libsais_sys::{libsais, libsais16, libsais16x64, libsais64};

use crate::context::{self, SaisContext};

macro_rules! ctx_fn_or_unimplemented {
    ($mod_name:ident, unimplemented, $($parameter:ident),*) => {
        unimplemented!("This function is currently not implemented by {}.", stringify!($mod_name))
    };
    ($mod_name:ident, $ctx_fn:ident, $($parameter:ident),*) => {
        $mod_name::$ctx_fn(
            $($parameter),*
        )
    };
}

macro_rules! fn_with_or_without_threads {
    ($mod_name:ident, $libsais_fn:ident, $p0:ident, $p1:ident,$p2:ident,$p3:ident,$p4:ident,$p5:ident, all()) => {
        $mod_name::$libsais_fn($p0, $p1, $p2, $p3, $p4)
    };
    ($mod_name:ident, $libsais_fn:ident, $p0:ident, $p1:ident,$p2:ident,$p3:ident,$p4:ident,$p5:ident, feature = "openmp") => {
        $mod_name::$libsais_fn($p0, $p1, $p2, $p3, $p4, $p5)
    };
}

macro_rules! libsais_functions_impl {
    (
        $struct_name:ident,
        $input_type:ty,
        $output_type:ty,
        $libsais_mod:ident,
        $libsais_fn:ident,
        $libsais_gsa_fn:ident,
        $libsais_ctx_fn:ident,
        $libsais_gsa_ctx_fn:ident,
        $($parallelism_tail:tt)+
    ) => {
        #[cfg($($parallelism_tail)+)]
        pub struct $struct_name {}

        #[cfg($($parallelism_tail)+)]
        impl LibsaisFunctions<$input_type, $output_type> for $struct_name {
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
                            frequency_table_ptr,
                            _num_threads,
                            $($parallelism_tail)+
                        ),
                        (true, Some(_context)) => {
                            ctx_fn_or_unimplemented!(
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
                            frequency_table_ptr,
                            _num_threads,
                            $($parallelism_tail)+
                        )
                        ,
                        (false, Some(_context)) => {
                            ctx_fn_or_unimplemented!(
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
        }
    };
}

libsais_functions_impl!(
    SingleThreaded8Input32Output,
    u8,
    i32,
    libsais,
    libsais,
    libsais_gsa,
    libsais_ctx,
    libsais_gsa_ctx,
    all()
);

libsais_functions_impl!(
    SingleThreaded8Input64Output,
    u8,
    i64,
    libsais64,
    libsais64,
    libsais64_gsa,
    unimplemented,
    unimplemented,
    all()
);

libsais_functions_impl!(
    SingleThreaded16Input32Output,
    u16,
    i32,
    libsais16,
    libsais16,
    libsais16_gsa,
    libsais16_ctx,
    libsais16_gsa_ctx,
    all()
);

libsais_functions_impl!(
    SingleThreaded16Input64Output,
    u16,
    i64,
    libsais16x64,
    libsais16x64,
    libsais16x64_gsa,
    unimplemented,
    unimplemented,
    all()
);

libsais_functions_impl!(
    MultiThreaded8Input32Output,
    u8,
    i32,
    libsais,
    libsais_omp,
    libsais_gsa_omp,
    unimplemented,
    unimplemented,
    feature = "openmp"
);

libsais_functions_impl!(
    MultiThreaded8Input64Output,
    u8,
    i64,
    libsais64,
    libsais64_omp,
    libsais64_gsa_omp,
    unimplemented,
    unimplemented,
    feature = "openmp"
);

libsais_functions_impl!(
    MultiThreaded16Input32Output,
    u16,
    i32,
    libsais16,
    libsais16_omp,
    libsais16_gsa_omp,
    unimplemented,
    unimplemented,
    feature = "openmp"
);

libsais_functions_impl!(
    MultiThreaded16Input64Output,
    u16,
    i64,
    libsais16x64,
    libsais16x64_omp,
    libsais16x64_gsa_omp,
    unimplemented,
    unimplemented,
    feature = "openmp"
);

#[cfg(true)]
pub trait LibsaisFunctions<I: InputBits, O: OutputBits> {
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
}

// struct SingleThreaded32Input32Output {}
// struct SingleThreaded64Input64Output {}

// #[cfg(feature = "openmp")]
// struct MultiThreaded32Input32Output {}
// #[cfg(feature = "openmp")]
// struct MultiThreaded64Input64Output {}

// -------------------- Parallelism and implementations --------------------
pub trait Parallelism {
    type Context: SaisContext;
    type WithInput<I: InputBits, O: OutputBits>: InputDispatch<I, O>;
}

pub enum SingleThreaded {}

impl Parallelism for SingleThreaded {
    type Context = context::SingleThreadedSaisContext;
    type WithInput<I: InputBits, O: OutputBits> = SingleThreadedInputDispatcher<I, O>;
}

#[cfg(feature = "openmp")]
pub enum MultiThreaded {}

#[cfg(feature = "openmp")]
impl Parallelism for MultiThreaded {
    type Context = context::MultiThreadedSaisContext;
    type WithInput<I: InputBits, O: OutputBits> = MultiThreadedInputDispatcher<I, O>;
}

// -------------------- Typestate traits for Builder API --------------------
pub trait Input: sealed::Sealed {}

pub trait Output: sealed::Sealed {}

pub enum Undecided {}

impl sealed::Sealed for Undecided {}

impl Input for Undecided {}

impl Output for Undecided {}

// -------------------- InputBits and OutputBits with implementations for u8, u16, i32, i64 --------------------
pub trait InputBits: sealed::Sealed + Into<i64> + Clone {
    type SingleThreadedOutputDispatcher<O: OutputBits>: OutputDispatch<Self, O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputBits>: OutputDispatch<Self, O>;
}

impl<I: InputBits> Input for I {}

pub trait OutputBits:
    sealed::Sealed + TryFrom<usize, Error: std::fmt::Debug> + Into<i64> + Clone
{
    type SingleThreaded8InputFunctions: LibsaisFunctions<u8, Self>;
    type SingleThreaded16InputFunctions: LibsaisFunctions<u16, Self>;
    // type SingleThreaded32InputFunctions: LibsaisFunctions<i32, Self>;
    // type SingleThreaded64InputFunctions: LibsaisFunctions<i64, Self>;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions: LibsaisFunctions<u8, Self>;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions: LibsaisFunctions<u16, Self>;
    // #[cfg(feature = "openmp")]
    // type MultiThreaded32InputFunctions: LibsaisFunctions<i32, Self>;
    // #[cfg(feature = "openmp")]
    // type MultiThreaded64InputFunctions: LibsaisFunctions<i64, Self>;
}

impl<B: OutputBits> Output for B {}

impl sealed::Sealed for u8 {}

impl InputBits for u8 {
    type SingleThreadedOutputDispatcher<O: OutputBits> = SingleThreaded8InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputBits> = MultiThreaded8InputOutputDispatcher<O>;
}

impl sealed::Sealed for u16 {}

impl InputBits for u16 {
    type SingleThreadedOutputDispatcher<O: OutputBits> = SingleThreaded16InputOutputDispatcher<O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputBits> = MultiThreaded16InputOutputDispatcher<O>;
}

impl sealed::Sealed for i32 {}

// TODO implement with correct types and think about mutability
// impl InputBits for i32 {
//     type SingleThreadedOutputDispatcher<O: OutputBits> = SingleThreaded32InputOutputDispatcher<O>;
//     #[cfg(feature = "openmp")]
//     type MultiThreadedOutputDispatcher<O: OutputBits> = SingleThreaded32InputOutputDispatcher<O>;
// }

impl OutputBits for i32 {
    type SingleThreaded8InputFunctions = SingleThreaded8Input32Output;
    type SingleThreaded16InputFunctions = SingleThreaded16Input32Output;
    // type SingleThreaded32InputFunctions = SingleThreaded32Input32Output;
    // type SingleThreaded64InputFunctions = SingleThreaded64Input32Output;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions = MultiThreaded8Input32Output;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions = MultiThreaded16Input32Output;
    // #[cfg(feature = "openmp")]
    // type MultiThreaded32InputFunctions = MultiThreaded32Input32Output;
    // #[cfg(feature = "openmp")]
    // type MultiThreaded64InputFunctions = MultiThreaded64Input32Output;
}

impl sealed::Sealed for i64 {}

// TODO implement with correct types and think about mutability
// impl InputBits for i64 {
//     type SingleThreadedOutputDispatcher<O: OutputBits> = SingleThreaded64InputOutputDispatcher<O>;
//     #[cfg(feature = "openmp")]
//     type MultiThreadedOutputDispatcher<O: OutputBits> = MultiThreaded64InputOutputDispatcher<O>;
// }

impl OutputBits for i64 {
    type SingleThreaded8InputFunctions = SingleThreaded8Input64Output;
    type SingleThreaded16InputFunctions = SingleThreaded16Input64Output;
    // type SingleThreaded32InputFunctions = SingleThreaded32Input64Output;
    // type SingleThreaded64InputFunctions = SingleThreaded64Input64Output;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions = MultiThreaded8Input64Output;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions = MultiThreaded16Input64Output;
    // #[cfg(feature = "openmp")]
    // type MultiThreaded32InputFunctions = MultiThreaded32Input64Output;
    // #[cfg(feature = "openmp")]
    // type MultiThreaded64InputFunctions = MultiThreaded64Input64Output;
}

// -------------------- InputDispatch and implementations --------------------
pub trait InputDispatch<I: InputBits, O: OutputBits> {
    type WithOutput: OutputDispatch<I, O>;
}

pub struct SingleThreadedInputDispatcher<I, O> {
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

impl<I: InputBits, O: OutputBits> InputDispatch<I, O> for SingleThreadedInputDispatcher<I, O> {
    type WithOutput = I::SingleThreadedOutputDispatcher<O>;
}

#[cfg(feature = "openmp")]
pub struct MultiThreadedInputDispatcher<I, O> {
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<I: InputBits, O: OutputBits> InputDispatch<I, O> for MultiThreadedInputDispatcher<I, O> {
    type WithOutput = I::MultiThreadedOutputDispatcher<O>;
}

// -------------------- OutputDispatch and implementations --------------------
pub trait OutputDispatch<I: InputBits, O: OutputBits> {
    type Functions: LibsaisFunctions<I, O>;
}

pub struct SingleThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O: OutputBits> OutputDispatch<u8, O> for SingleThreaded8InputOutputDispatcher<O> {
    type Functions = O::SingleThreaded8InputFunctions;
}

pub struct SingleThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O: OutputBits> OutputDispatch<u16, O> for SingleThreaded16InputOutputDispatcher<O> {
    type Functions = O::SingleThreaded16InputFunctions;
}

// pub struct SingleThreaded32InputOutputDispatcher<O> {
//     _output_marker: PhantomData<O>,
// }

// impl<O: OutputBits> OutputDispatch<i32, O> for SingleThreaded32InputOutputDispatcher<O> {
//     type Functions = O::SingleThreaded32InputFunctions;
// }

// pub struct SingleThreaded64InputOutputDispatcher<O> {
//     _output_marker: PhantomData<O>,
// }

// impl<O: OutputBits> OutputDispatch<i64, O> for SingleThreaded64InputOutputDispatcher<O> {
//     type Functions = O::SingleThreaded64InputFunctions;
// }

#[cfg(feature = "openmp")]
pub struct MultiThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O: OutputBits> OutputDispatch<u8, O> for MultiThreaded8InputOutputDispatcher<O> {
    type Functions = O::MultiThreaded8InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O: OutputBits> OutputDispatch<u16, O> for MultiThreaded16InputOutputDispatcher<O> {
    type Functions = O::MultiThreaded16InputFunctions;
}

mod sealed {
    pub trait Sealed {}
}
