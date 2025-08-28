mod impls;

use std::{ffi::c_void, marker::PhantomData};

use crate::{
    InputElement,
    sealed::Sealed,
    typestate::{OutputElementOrUndecided, Parallelism, ParallelismOrUndecided},
};

// These types are the key gadgets that allow the transition from the generic interface of this wrapper
// to the flat function interface of the C library.
// They essentially form a tree in the type system with associated types as edge transitions.
// Maybe it is possible topimplement this goal more in a more simple way, but this is what I came up with.
pub type SmallAlphabetFunctionsDispatch<I, O, P> =
    <<<P as Parallelism>::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<
        I,
        O,
    >>::SmallAlphabetFunctions;

pub type LargeAlphabetFunctionsDispatch<I, O, P> =
    <<<P as Parallelism>::WithInput<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<
        I,
        O,
    >>::LargeAlphabetFunctions;

pub type LcpFunctionsDispatch<I, O, P> = <<<P as Parallelism>::WithInput<I, O> as InputDispatch<
    I,
    O,
>>::WithOutput as OutputDispatch<I, O>>::LcpFunctions;

pub type SmallAlphabetFunctionsDispatchOrUnimplemented<I, O, P> =
    <<<P as ParallelismOrUndecided>::WithInputOrUnimplemented<I, O> as InputDispatch<I, O>>::WithOutput as OutputDispatch<
        I,
        O,
    >>::SmallAlphabetFunctions;

// -------------------- traits that model libsais functions --------------------
#[allow(clippy::too_many_arguments)]
pub trait LibsaisFunctionsSmallAlphabet<I: InputElement, O: OutputElementOrUndecided>:
    Sealed
{
    unsafe fn libsais(
        text_ptr: *const I,
        suffix_array_buffer_ptr: *mut O,
        text_len: O,
        extra_space: O,
        frequency_table_ptr: *mut O,
        num_threads: O,
        generalized_suffix_array: bool,
        context: Option<*mut c_void>,
    ) -> O;

    unsafe fn libsais_bwt(
        text_ptr: *const I,
        bwt_buffer_ptr: *mut I,
        suffix_array_buffer_ptr: *mut O,
        text_len: O,
        extra_space: O,
        frequency_table_ptr: *mut O,
        num_threads: O,
        context: Option<*mut c_void>,
    ) -> O;

    unsafe fn libsais_bwt_aux(
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

    unsafe fn libsais_create_ctx(num_threads: O) -> *mut c_void;
    unsafe fn libsais_free_ctx(ctx: *mut c_void);

    unsafe fn libsais_unbwt(
        bwt_ptr: *const I,
        text_buffer_ptr: *mut I,
        suffix_array_buffer_ptr: *mut O,
        bwt_len: O,
        frequency_table_ptr: *const O,
        primary_index: O,
        num_threads: O,
        context: Option<*mut c_void>,
    ) -> O;

    unsafe fn libsais_unbwt_aux(
        bwt_ptr: *const I,
        text_buffer_ptr: *mut I,
        suffix_array_buffer_ptr: *mut O,
        bwt_len: O,
        frequency_table_ptr: *const O,
        aux_indices_sampling_rate: O,
        aux_indices_buffer_ptr: *const O,
        num_threads: O,
        context: Option<*mut c_void>,
    ) -> O;

    unsafe fn libsais_unbwt_create_ctx(num_threads: O) -> *mut c_void;
    unsafe fn libsais_unbwt_free_ctx(ctx: *mut c_void);
}

pub trait LibsaisFunctionsLargeAlphabet<I: InputElement, O: OutputElementOrUndecided>:
    Sealed
{
    unsafe fn libsais_large_alphabet(
        text_ptr: *mut I,
        suffix_array_buffer_ptr: *mut O,
        text_len: O,
        alphabet_size: O,
        extra_space: O,
        num_threads: O,
    ) -> O;
}

pub trait LibsaisLcpFunctions<I: InputElement, O: OutputElementOrUndecided>: Sealed {
    unsafe fn libsais_plcp(
        text_ptr: *const I,
        suffix_array_ptr: *const O,
        plcp_ptr: *mut O,
        text_len: O,
        num_threads: O,
        generalized_suffix_array: bool,
    ) -> O;

    unsafe fn libsais_lcp(
        plcp_ptr: *const O,
        suffix_array_ptr: *const O,
        lcp_ptr: *mut O,
        suffix_array_len: O,
        num_threads: O,
    ) -> O;
}

// -------------------- placeholder type for output dispatch when fucntions are unimplemented --------------------
pub struct FunctionsUnimplemented {}

impl Sealed for FunctionsUnimplemented {}

impl<I: InputElement, O: OutputElementOrUndecided> LibsaisFunctionsSmallAlphabet<I, O>
    for FunctionsUnimplemented
{
    unsafe fn libsais(
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

    unsafe fn libsais_bwt(
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

    unsafe fn libsais_bwt_aux(
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

    unsafe fn libsais_create_ctx(_num_threads: O) -> *mut c_void {
        unimplemented!(
            "There is no libsais create ctx implementation for this combination of input and output types.",
        );
    }

    unsafe fn libsais_free_ctx(_ctx: *mut c_void) {
        unimplemented!(
            "There is no libsais free ctx implementation for this combination of input and output types.",
        );
    }

    unsafe fn libsais_unbwt(
        _bwt_ptr: *const I,
        _text_ptr: *mut I,
        _suffix_array_buffer_ptr: *mut O,
        _bwt_len: O,
        _frequency_table_ptr: *const O,
        _primary_index: O,
        _num_threads: O,
        _context: Option<*mut c_void>,
    ) -> O {
        unimplemented!(
            "There is no libsais unbwt implementation for this combination of input and output types.",
        );
    }

    unsafe fn libsais_unbwt_aux(
        _bwt_ptr: *const I,
        _text_ptr: *mut I,
        _suffix_array_buffer_ptr: *mut O,
        _bwt_len: O,
        _frequency_table_ptr: *const O,
        _aux_indices_sampling_rate: O,
        _aux_indices_buffer_ptr: *const O,
        _num_threads: O,
        _context: Option<*mut c_void>,
    ) -> O {
        unimplemented!(
            "There is no libsais unbwt aux implementation for this combination of input and output types.",
        );
    }

    unsafe fn libsais_unbwt_create_ctx(_num_threads: O) -> *mut c_void {
        unimplemented!(
            "There is no libsais unbwt create ctx implementation for this combination of input and output types.",
        );
    }

    unsafe fn libsais_unbwt_free_ctx(_ctx: *mut c_void) {
        unimplemented!(
            "There is no libsais unbwt free ctx implementation for this combination of input and output types.",
        );
    }
}

impl<I: InputElement, O: OutputElementOrUndecided> LibsaisFunctionsLargeAlphabet<I, O>
    for FunctionsUnimplemented
{
    unsafe fn libsais_large_alphabet(
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

impl<I: InputElement, O: OutputElementOrUndecided> LibsaisLcpFunctions<I, O>
    for FunctionsUnimplemented
{
    unsafe fn libsais_plcp(
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

    unsafe fn libsais_lcp(
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

// -------------------- InputDispatch and implementations --------------------
pub trait InputDispatch<I: InputElement, O: OutputElementOrUndecided>: Sealed {
    type WithOutput: OutputDispatch<I, O>;
}

pub struct SingleThreadedInputDispatcher<I, O> {
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

impl<I, O> Sealed for SingleThreadedInputDispatcher<I, O> {}

impl<I: InputElement, O: OutputElementOrUndecided> InputDispatch<I, O>
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
impl<I, O> Sealed for MultiThreadedInputDispatcher<I, O> {}

#[cfg(feature = "openmp")]
impl<I: InputElement, O: OutputElementOrUndecided> InputDispatch<I, O>
    for MultiThreadedInputDispatcher<I, O>
{
    type WithOutput = I::MultiThreadedOutputDispatcher<O>;
}

pub struct UnimplementedInputDispatcher<I, O> {
    _input_marker: PhantomData<I>,
    _output_marker: PhantomData<O>,
}

impl<I, O> Sealed for UnimplementedInputDispatcher<I, O> {}

impl<I: InputElement, O: OutputElementOrUndecided> InputDispatch<I, O>
    for UnimplementedInputDispatcher<I, O>
{
    type WithOutput = UnimplementedOutputDispatcher<O>;
}

// -------------------- OutputDispatch and implementations --------------------
pub trait OutputDispatch<I: InputElement, O: OutputElementOrUndecided>: Sealed {
    type SmallAlphabetFunctions: LibsaisFunctionsSmallAlphabet<I, O>;
    type LargeAlphabetFunctions: LibsaisFunctionsLargeAlphabet<I, O>;
    type LcpFunctions: LibsaisLcpFunctions<I, O>;
}

pub struct SingleThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> Sealed for SingleThreaded8InputOutputDispatcher<O> {}

impl<O: OutputElementOrUndecided> OutputDispatch<u8, O>
    for SingleThreaded8InputOutputDispatcher<O>
{
    type SmallAlphabetFunctions = O::SingleThreaded8InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::SingleThreaded8InputFunctions;
}

pub struct SingleThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> Sealed for SingleThreaded16InputOutputDispatcher<O> {}

impl<O: OutputElementOrUndecided> OutputDispatch<u16, O>
    for SingleThreaded16InputOutputDispatcher<O>
{
    type SmallAlphabetFunctions = O::SingleThreaded16InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::SingleThreaded16InputFunctions;
}

pub struct SingleThreaded32InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> Sealed for SingleThreaded32InputOutputDispatcher<O> {}

impl<O: OutputElementOrUndecided> OutputDispatch<i32, O>
    for SingleThreaded32InputOutputDispatcher<O>
{
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::SingleThreaded32InputFunctions;
    type LcpFunctions = O::SingleThreaded32InputFunctions;
}

pub struct SingleThreaded64InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> Sealed for SingleThreaded64InputOutputDispatcher<O> {}

impl<O: OutputElementOrUndecided> OutputDispatch<i64, O>
    for SingleThreaded64InputOutputDispatcher<O>
{
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::SingleThreaded64InputFunctions;
    type LcpFunctions = O::SingleThreaded64InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded8InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> Sealed for MultiThreaded8InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementOrUndecided> OutputDispatch<u8, O> for MultiThreaded8InputOutputDispatcher<O> {
    type SmallAlphabetFunctions = O::MultiThreaded8InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::MultiThreaded8InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded16InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> Sealed for MultiThreaded16InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementOrUndecided> OutputDispatch<u16, O>
    for MultiThreaded16InputOutputDispatcher<O>
{
    type SmallAlphabetFunctions = O::MultiThreaded16InputFunctions;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = O::MultiThreaded16InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded32InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> Sealed for MultiThreaded32InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementOrUndecided> OutputDispatch<i32, O>
    for MultiThreaded32InputOutputDispatcher<O>
{
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::MultiThreaded32InputFunctions;
    type LcpFunctions = O::MultiThreaded32InputFunctions;
}

#[cfg(feature = "openmp")]
pub struct MultiThreaded64InputOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

#[cfg(feature = "openmp")]
impl<O> Sealed for MultiThreaded64InputOutputDispatcher<O> {}

#[cfg(feature = "openmp")]
impl<O: OutputElementOrUndecided> OutputDispatch<i64, O>
    for MultiThreaded64InputOutputDispatcher<O>
{
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = O::MultiThreaded64InputFunctions;
    type LcpFunctions = O::MultiThreaded64InputFunctions;
}

pub struct UnimplementedOutputDispatcher<O> {
    _output_marker: PhantomData<O>,
}

impl<O> Sealed for UnimplementedOutputDispatcher<O> {}

impl<I: InputElement, O: OutputElementOrUndecided> OutputDispatch<I, O>
    for UnimplementedOutputDispatcher<O>
{
    type SmallAlphabetFunctions = FunctionsUnimplemented;
    type LargeAlphabetFunctions = FunctionsUnimplemented;
    type LcpFunctions = FunctionsUnimplemented;
}
