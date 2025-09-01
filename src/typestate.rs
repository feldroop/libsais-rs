/*!
 * Typestate model for builder APIs, most likely not relevant to you.
 */

use either::Either;

use crate::{
    InputElement, OutputElement,
    generics_dispatch::{
        FunctionsUnimplemented, InputDispatch, LibsaisFunctionsLargeAlphabet,
        LibsaisFunctionsSmallAlphabet, LibsaisLcpFunctions, SingleThreadedInputDispatcher,
        UnimplementedInputDispatcher,
    },
    sealed::Sealed,
};

#[cfg(feature = "openmp")]
use crate::generics_dispatch::MultiThreadedInputDispatcher;

pub enum Undecided {}

impl Sealed for Undecided {}

pub trait ParallelismOrUndecided: Sealed {
    type WithInputOrUnimplemented<I: InputElement, O: OutputElementOrUndecided>: InputDispatch<I, O>;
}

impl ParallelismOrUndecided for Undecided {
    type WithInputOrUnimplemented<I: InputElement, O: OutputElementOrUndecided> =
        UnimplementedInputDispatcher<I, O>;
}

impl<P: Parallelism> ParallelismOrUndecided for P {
    type WithInputOrUnimplemented<I: InputElement, O: OutputElementOrUndecided> =
        P::WithInput<I, O>;
}

/// Decision about whether the OpenMP based functions of `libsais` will be used. [`MultiThreaded`] is
/// only available when the crate feature `openmp` is activated.
pub trait Parallelism: Sealed {
    type WithInput<I: InputElement, O: OutputElementOrUndecided>: InputDispatch<I, O>;
}

pub enum SingleThreaded {}

impl Sealed for SingleThreaded {}

impl Parallelism for SingleThreaded {
    type WithInput<I: InputElement, O: OutputElementOrUndecided> =
        SingleThreadedInputDispatcher<I, O>;
}

#[cfg(feature = "openmp")]
pub enum MultiThreaded {}

#[cfg(feature = "openmp")]
impl Sealed for MultiThreaded {}

#[cfg(feature = "openmp")]
impl Parallelism for MultiThreaded {
    type WithInput<I: InputElement, O: OutputElementOrUndecided> =
        MultiThreadedInputDispatcher<I, O>;
}

// all of the complexity of this trait is only needed for the the Drop implementation for
// the context types.
pub trait OutputElementOrUndecided: Sealed + Sized {
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

impl OutputElementOrUndecided for Undecided {
    type SingleThreaded8InputFunctions = FunctionsUnimplemented;
    type SingleThreaded16InputFunctions = FunctionsUnimplemented;
    type SingleThreaded32InputFunctions = FunctionsUnimplemented;
    type SingleThreaded64InputFunctions = FunctionsUnimplemented;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions = FunctionsUnimplemented;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions = FunctionsUnimplemented;
    #[cfg(feature = "openmp")]
    type MultiThreaded32InputFunctions = FunctionsUnimplemented;
    #[cfg(feature = "openmp")]
    type MultiThreaded64InputFunctions = FunctionsUnimplemented;
}

impl<O: OutputElement> OutputElementOrUndecided for O {
    type SingleThreaded8InputFunctions = <O as OutputElement>::SingleThreaded8InputFunctions;
    type SingleThreaded16InputFunctions = <O as OutputElement>::SingleThreaded16InputFunctions;
    type SingleThreaded32InputFunctions = <O as OutputElement>::SingleThreaded32InputFunctions;
    type SingleThreaded64InputFunctions = <O as OutputElement>::SingleThreaded64InputFunctions;

    #[cfg(feature = "openmp")]
    type MultiThreaded8InputFunctions = <O as OutputElement>::MultiThreaded8InputFunctions;
    #[cfg(feature = "openmp")]
    type MultiThreaded16InputFunctions = <O as OutputElement>::MultiThreaded16InputFunctions;
    #[cfg(feature = "openmp")]
    type MultiThreaded32InputFunctions = <O as OutputElement>::MultiThreaded32InputFunctions;
    #[cfg(feature = "openmp")]
    type MultiThreaded64InputFunctions = <O as OutputElement>::MultiThreaded64InputFunctions;
}

pub trait BufferModeOrUndecided: Sealed {}

impl BufferModeOrUndecided for Undecided {}

/// Decision about whether an owned [`Vec`]-based buffer or a user-provided slice-based buffer is used.
pub trait BufferMode: Sealed {
    type Buffer<'a, T: 'a>;

    fn buffer_to_either<'a, T>(buffer: Self::Buffer<'a, T>) -> Either<Vec<T>, &'a mut [T]>;

    fn either_to_buffer<'a, T: std::fmt::Debug>(
        either_: Either<Vec<T>, &'a mut [T]>,
    ) -> Self::Buffer<'a, T>;

    fn unwrap_or_allocate<'a, T, F>(opt: Option<&'a mut [T]>, f: F) -> Self::Buffer<'a, T>
    where
        F: FnOnce() -> Vec<T>;
}

impl<B: BufferMode> BufferModeOrUndecided for B {}

pub struct BorrowedBuffer {}

impl Sealed for BorrowedBuffer {}

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
        F: FnOnce() -> Vec<T>,
    {
        opt.unwrap()
    }
}

pub struct OwnedBuffer {}

impl Sealed for OwnedBuffer {}

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
        F: FnOnce() -> Vec<T>,
    {
        f()
    }
}

/// In some operations (BWT, BWT reversal and LCP), the input can also be used as the output buffer.
pub trait BufferModeOrReplaceInput: Sealed {}

impl<B: BufferMode> BufferModeOrReplaceInput for B {}

pub struct ReplaceInput {}

impl Sealed for ReplaceInput {}

impl BufferModeOrReplaceInput for ReplaceInput {}

/// Decision about whether auxiliary indices are additionally returned during BWT construction. See
/// [`bwt`](super::bwt) for details.
pub trait AuxIndicesMode: Sealed {}

pub struct NoAuxIndices {}

impl Sealed for NoAuxIndices {}

impl AuxIndicesMode for NoAuxIndices {}

impl<B: BufferMode> AuxIndicesMode for B {}
