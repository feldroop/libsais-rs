use either::Either;

use std::marker::PhantomData;

use crate::type_state::BufferMode;

// For internal use in this library. This struct is used to be agnostic over the usage of an owned buffer
// or a borrowed slice. When a Vec is used, `B` should always be OwnedBuffer and when a slice is used,
// B should always be a BorrowedBuffer
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OwnedOrBorrowed<'a, T, B> {
    pub(crate) buffer: Either<Vec<T>, &'a mut [T]>,
    pub(crate) _marker: PhantomData<B>,
}

impl<'a, T: std::fmt::Debug, B: BufferMode> OwnedOrBorrowed<'a, T, B> {
    pub(crate) fn new(buffer: B::Buffer<'a, T>) -> OwnedOrBorrowed<'a, T, B> {
        OwnedOrBorrowed {
            buffer: B::buffer_to_either(buffer),
            _marker: PhantomData,
        }
    }

    pub(crate) fn into_inner(self) -> B::Buffer<'a, T> {
        B::either_to_buffer(self.buffer)
    }
}

impl<'a, T, B: BufferMode> OwnedOrBorrowed<'a, T, B> {
    pub(crate) fn shorten_buffer_to(&mut self, len: usize) {
        match &mut self.buffer {
            Either::Left(owned) => {
                owned.truncate(len);
                owned.shrink_to_fit();
            }
            Either::Right(borrowed) => *borrowed = &mut std::mem::take(borrowed)[..len],
        }
    }
}

impl<'a, T, B: BufferMode> OwnedOrBorrowed<'a, T, B>
where
    T: TryFrom<usize, Error: std::fmt::Debug> + Clone + std::fmt::Debug,
{
    pub(crate) fn take_buffer_or_allocate<F: FnOnce() -> Vec<T>>(
        opt: Option<&'a mut [T]>,
        f: F,
    ) -> OwnedOrBorrowed<'a, T, B> {
        Self::new(B::unwrap_or_allocate(opt, f))
    }
}
