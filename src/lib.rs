pub mod bwt;
pub mod context;
pub mod error;
pub mod lcp;
pub mod plcp;
pub mod suffix_array;
pub mod type_state;
pub mod unbwt;

mod generics_dispatch;
mod owned_or_borrowed;

use generics_dispatch::{
    LibsaisFunctionsLargeAlphabet, LibsaisFunctionsSmallAlphabet, LibsaisLcpFunctions,
    OutputDispatch,
};
use sealed::Sealed;

/// The version of the C library libsais wrapped by this crate
pub use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

/// The maximum text size that this library can handle when using i32-based buffers
pub const LIBSAIS_I32_OUTPUT_MAXIMUM_SIZE: usize = 2147483647;

/// The maximum text size that this library can handle when using i64-based buffers
pub const LIBSAIS_I64_OUTPUT_MAXIMUM_SIZE: usize = 9223372036854775807;

#[doc(inline)]
pub use {
    bwt::BwtConstruction, lcp::LcpConstruction, plcp::PlcpConstruction,
    suffix_array::SuffixArrayConstruction, unbwt::UnBwt,
};

// TODOs:

//      derives of public types

//      small benchmarks
//      setup CI and such
//      good docs and examples, README (figure out good pub exports)
//      release-plz good release (also libsais-sys update)

//      wait for answer on ilya grebnov context question
//      figure out whether to use ParallelismUndebiced or no Parallelism at all
//      fix context drop/undecided issue in drop (also TODOs in safe comment)
//      make context sound via forcing parallelism decision before supplying context (probably for all contructions)
//      when context is there, could get rid of parallelism marker
//      context in into other marker type

pub trait InputElement:
    Sealed + std::fmt::Debug + Copy + TryFrom<usize, Error: std::fmt::Debug> + Into<i64> + Clone + Ord
{
    const RECOMMENDED_EXTRA_SPACE: usize;
    const ZERO: Self;

    type SingleThreadedOutputDispatcher<O: OutputElement>: OutputDispatch<Self, O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElement>: OutputDispatch<Self, O>;
}

pub trait OutputElement:
    Sealed
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

pub trait SmallAlphabet: InputElement {
    const FREQUENCY_TABLE_SIZE: usize;
}

pub trait LargeAlphabet: InputElement + OutputElement {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ThreadCount {
    pub(crate) value: u16,
}

impl ThreadCount {
    pub const fn fixed(thread_count: u16) -> Self {
        if thread_count == 0 {
            panic!("Fixed thread count cannot be 0");
        }

        Self {
            value: thread_count,
        }
    }

    pub const fn openmp_default() -> Self {
        Self { value: 0 }
    }
}

mod sealed {
    pub trait Sealed {}
}
