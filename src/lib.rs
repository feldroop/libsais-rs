/*!
 * An idiomatic and mostly safe API wrapper for the awesome and _very_ fast C library
 * [`libsais`] by Ilya Grebnov.
 *
 * [`libsais`] provides highly optimized implementation for the features listed below.
 * These implementations are widely used in the analysis of very large sequences, such as genome analysis
 * in bioinformatics. For example, [`libsais`] contains the fastest suffix array construction
 * implementation to date. [Comparison to `libdivsufsort`] | [Benchmark crates.io]
 *
 * <div class="warning">
 *
 * This crate is not yet battle-tested, there might be bugs. The API is still subject to small changes.
 * Any kind of feedback and suggestions via the issue tracker is highly appreciated!
 *
 * </div>
 *
 * ## Features
 *
 * This crate exposes the whole functionality of [`libsais`].
 * It might be useful to also check out the [documentation of the original library].
 * For further details on the individual features, please refer to the module-level documentation.
 *
 * * [suffix_array]: Construct suffix arrays for `u8`/`u16`/`i32`/`i64` input texts and `i32`/`i64` output texts,
 *   with support for generalized suffix arrays
 * * [bwt]: Construct the Burrows-Wheeler-Transform (BWT) for `u8`/`u16` texts
 * * [unbwt]: Recover the original text from a BWT
 * * [plcp]: Construct the permuted longest common prefix array (PLCP) for a suffix array and text
 * * [lcp]: Construct the longest common prefix array from a PLCP for a suffix array and text
 * * [context]: Use a memory allocation optimization for repeated calls on small inputs
 *
 * ## Usage
 *
 * This crate provides generic builder-like APIs for all of the features listed above.
 * The following is a simple example of how to use this library to construct a suffix array in parallel:
 *
 * ```rust
 * use libsais::{SuffixArrayConstruction, ThreadCount};
 *
 * let text = b"barnabasbabblesaboutbananas";
 * let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
 *     .in_owned_buffer()
 *     .multi_threaded(ThreadCount::openmp_default())
 *     .run()
 *     .expect("The example on the front page should really work")
 *     .into_vec();
 * ```
 *
 * ## Examples
 *
 * There are [examples] for multiple non-trivial use-cases of this library.
 *
 * ## Multithreading
 *
 * This library supports the multithreaded implementations of [`libsais`] via the optional `openmp` feature,
 * which is enabled by default.
 *
 * [`libsais`]: https://github.com/IlyaGrebnov/libsais
 * [Comparison to `libdivsufsort`]: https://github.com/IlyaGrebnov/libsais/blob/master/Benchmarks.md
 * [Benchmark crates.io]: https://github.com/feldroop/benchmark_crates_io_sacas
 * [documentation of the original library]: https://github.com/IlyaGrebnov/libsais
 * [examples]: https://github.com/feldroop/libsais-rs/tree/master/examples
 */

/// Construct a Burrows-Wheeler-Transform
pub mod bwt;

/// Use an optimization for repeated calls of `libsais` functions on small inputs
pub mod context;

/// The error type of this crate
pub mod error;

/// Construct the permuted longest common prefix array
pub mod lcp;

/// Construct the permuted longest common prefix array
pub mod plcp;

/// Construct a (generalized) suffix array
pub mod suffix_array;

/// Typestate for builder APIs, ikely not relevant for you
pub mod type_state;

/// Recover the text from a Burrows-Wheeler-Transform
pub mod unbwt;

mod generics_dispatch;
mod owned_or_borrowed;

use generics_dispatch::{
    LibsaisFunctionsLargeAlphabet, LibsaisFunctionsSmallAlphabet, LibsaisLcpFunctions,
    OutputDispatch,
};
use sealed::Sealed;
use type_state::OutputElementOrUndecided;

/// The version of the original C library `libsais` wrapped by this crate
pub use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

/// The maximum text size that this library can handle when using `i32`-based buffers
pub const LIBSAIS_I32_OUTPUT_MAXIMUM_SIZE: usize = 2147483647;

/// The maximum text size that this library can handle when using `i64`-based buffers
pub const LIBSAIS_I64_OUTPUT_MAXIMUM_SIZE: usize = 9223372036854775807;

// Main entry points of the library
#[doc(inline)]
pub use {
    bwt::BwtConstruction, lcp::LcpConstruction, plcp::PlcpConstruction,
    suffix_array::SuffixArrayConstruction, unbwt::UnBwt,
};

// TODOs:

//      some comments in code
//      good docs functions

//      good docs module and crate level

//      setup CI and such, also without default features, clippy, etc.
//      release-plz good release (also libsais-sys update)

/// Possible element types of input texts and output data structures storing text elements implement this trait.
/// You cannot implement it and don't need to.
pub trait InputElement:
    Sealed + std::fmt::Debug + Copy + Clone + Ord + TryFrom<usize, Error: std::fmt::Debug> + Into<i64>
{
    const RECOMMENDED_EXTRA_SPACE: usize;
    const ZERO: Self;

    type SingleThreadedOutputDispatcher<O: OutputElementOrUndecided>: OutputDispatch<Self, O>;
    #[cfg(feature = "openmp")]
    type MultiThreadedOutputDispatcher<O: OutputElementOrUndecided>: OutputDispatch<Self, O>;
}

/// Possible element types of output data structures storing indices implement this trait.
/// You cannot implement it and don't need to.
pub trait OutputElement:
    Sealed
    + std::fmt::Debug
    + std::fmt::Display
    + Copy
    + Clone
    + TryFrom<usize, Error: std::fmt::Debug>
    + Into<i64>
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

/// Certain features such as the [Burrows-Wheeler-Transform](bwt) are only available for `u8`/`u16`-based texts.
pub trait SmallAlphabet: InputElement {
    const FREQUENCY_TABLE_SIZE: usize;
}

/// When using `i32`/`i64`-based texts, the API for suffix array construction changes slightly. [Details](TODO)
pub trait LargeAlphabet: InputElement + OutputElement {}

/// A wrapper type for passing the desired number of threads. Only useful when the `openmp` feature is activated.
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
