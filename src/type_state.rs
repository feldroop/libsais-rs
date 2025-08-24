use either::Either;

use crate::{
    InputElement, OutputElement,
    generics_dispatch::{InputDispatch, SingleThreadedInputDispatcher},
    sealed::Sealed,
};

#[cfg(feature = "openmp")]
use crate::generics_dispatch::MultiThreadedInputDispatcher;

pub enum Undecided {}

impl Sealed for Undecided {}

pub trait ParallelismOrUndecided: Sealed {
    // TODO fn free_ctx<I,O>
}

impl ParallelismOrUndecided for Undecided {}

impl<P: Parallelism> ParallelismOrUndecided for P {}

pub trait Parallelism: Sealed {
    type WithInput<I: InputElement, O: OutputElement>: InputDispatch<I, O>;
}

pub enum SingleThreaded {}

impl Sealed for SingleThreaded {}

impl Parallelism for SingleThreaded {
    type WithInput<I: InputElement, O: OutputElement> = SingleThreadedInputDispatcher<I, O>;
}

#[cfg(feature = "openmp")]
pub enum MultiThreaded {}

#[cfg(feature = "openmp")]
impl Sealed for MultiThreaded {}

#[cfg(feature = "openmp")]
impl Parallelism for MultiThreaded {
    type WithInput<I: InputElement, O: OutputElement> = MultiThreadedInputDispatcher<I, O>;
}

pub trait OutputElementOrUndecided: Sealed {}

impl OutputElementOrUndecided for Undecided {}

pub trait BufferModeOrUndecided: Sealed {}

impl BufferModeOrUndecided for Undecided {}

pub trait BufferMode: Sealed {
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
        T: TryFrom<usize, Error: std::fmt::Debug> + Clone,
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
        T: TryFrom<usize, Error: std::fmt::Debug> + Clone,
        F: FnOnce() -> Vec<T>,
    {
        f()
    }
}

pub trait BufferModeOrReplaceInput: Sealed {}

impl<B: BufferMode> BufferModeOrReplaceInput for B {}

pub struct ReplaceInput {}

impl Sealed for ReplaceInput {}

impl BufferModeOrReplaceInput for ReplaceInput {}

pub trait AuxIndicesMode: Sealed {}

pub struct NoAuxIndices {}

impl Sealed for NoAuxIndices {}

impl AuxIndicesMode for NoAuxIndices {}

impl<B: BufferMode> AuxIndicesMode for B {}

impl<B: OutputElement> OutputElementOrUndecided for B {}

pub trait IsValidOutputFor<I: InputElement>: Sealed + OutputElement {}

impl IsValidOutputFor<u8> for i32 {}
impl IsValidOutputFor<u16> for i32 {}
impl IsValidOutputFor<i32> for i32 {}

impl IsValidOutputFor<u8> for i64 {}
impl IsValidOutputFor<u16> for i64 {}
impl IsValidOutputFor<i64> for i64 {}

pub trait SupportsPlcpOutputFor<I: InputElement>: Sealed + OutputElement {}

impl SupportsPlcpOutputFor<u8> for i32 {}
impl SupportsPlcpOutputFor<u16> for i32 {}
impl SupportsPlcpOutputFor<i32> for i32 {}

impl SupportsPlcpOutputFor<u8> for i64 {}
impl SupportsPlcpOutputFor<u16> for i64 {}
