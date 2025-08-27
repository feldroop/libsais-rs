/*!
 * An idiomatic and mostly safe API wrapper for the awesome and _very_ fast C library
 * [`libsais`] by Ilya Grebnov.
 *
 * [`libsais`] provides highly optimized implementation for the features listed below.
 * These implementations are widely used in the analysis of very large sequences, such as genome analysis
 * in bioinformatics. For example, [`libsais`] contains the fastest implementation of a suffix array construction
 * algorithm to date. [Comparison to `libdivsufsort`] | [Benchmark crates.io]
 *
 * <div class="warning">
 *
 * This crate is not yet battle-tested, there might be bugs. The API is still subject to small changes.
 * Any kind of feedback and suggestions via the issue tracker is highly appreciated!
 *
 * </div>
 *
 * # Features
 *
 * This crate exposes the whole functionality of [`libsais`].
 * It might be useful to also check out the [documentation of the original library].
 *
 * * [`suffix_array`]: Construct suffix arrays for `u8`/`u16`/`i32`/`i64` input texts and `i32`/`i64` output texts,
 *   with support for generalized suffix arrays.
 * * [`bwt`]: Construct the Burrows-Wheeler-Transform (BWT) for `u8`/`u16` texts.
 * * [`unbwt`]: Recover the original text from a BWT.
 * * [`plcp`]: Construct the permuted longest common prefix array (PLCP) for a suffix array and text.
 * * [`lcp`]: Construct the longest common prefix array from a PLCP for a suffix array and text.
 * * [`context`]: Use a memory allocation optimization for repeated calls on small inputs.
 *
 * # Usage
 *
 * This crate provides generic builder-like APIs for all of the features listed above.
 * For further details on the individual features, please refer to the module-level documentation.
 * The API is a bit noisy due to lifetimes and typestate, so it is recommended to start with the
 * module-level documentation and [examples].
 *
 * The following is a simple example of how to use this library to construct a suffix array in parallel:
 *
 * ```
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
 * The primary entry points to this library are [`SuffixArrayConstruction`] and [`BwtConstruction`]. Obtaining
 * an [`LcpConstruction`], [`PlcpConstruction`] or [`UnBwt`] can only be done via an `unsafe` constructor or by
 * using the returned types of the primary operations. Passing logically wrong input to these secondary
 * functions of `libsais` can result in undefined behavior of the underlying C library.
 *
 * # Examples
 *
 * There are [examples] for multiple non-trivial use-cases of this library.
 *
 * # Multithreading
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

pub mod bwt;
pub mod context;
pub mod lcp;
pub mod plcp;
pub mod suffix_array;
pub mod typestate;
pub mod unbwt;

mod generics_dispatch;
mod owned_or_borrowed;

use std::fmt::{Debug, Display};

use generics_dispatch::{
    LibsaisFunctionsLargeAlphabet, LibsaisFunctionsSmallAlphabet, LibsaisLcpFunctions,
    OutputDispatch,
};
use sealed::Sealed;
use typestate::OutputElementOrUndecided;

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

/// When using `i32`/`i64`-based texts, the API for suffix array construction changes slightly.
/// See [`suffix_array`] for details.
pub trait LargeAlphabet: InputElement + OutputElement {}

/// Information about whether an [`OutputElement`] type can be used with a given [`InputElement`] type.
///
/// Notably, `i32` output supports only `i32` input and `i64` output supports only `i64` input.
pub trait IsValidOutputFor<I: InputElement>: Sealed + OutputElement {}

impl IsValidOutputFor<u8> for i32 {}
impl IsValidOutputFor<u16> for i32 {}
impl IsValidOutputFor<i32> for i32 {}

impl IsValidOutputFor<u8> for i64 {}
impl IsValidOutputFor<u16> for i64 {}
impl IsValidOutputFor<i64> for i64 {}

/// Information about whether an [`OutputElement`] type supports the construction of PLCP arrays.
///
/// Apart from the limitations of [`IsValidOutputFor`], `i64` output does't support PLCP construction
/// for `i64` input.
pub trait SupportsPlcpOutputFor<I: InputElement>:
    Sealed + OutputElement + IsValidOutputFor<I>
{
}

impl SupportsPlcpOutputFor<u8> for i32 {}
impl SupportsPlcpOutputFor<u16> for i32 {}
impl SupportsPlcpOutputFor<i32> for i32 {}

impl SupportsPlcpOutputFor<u8> for i64 {}
impl SupportsPlcpOutputFor<u16> for i64 {}

/// The error type of this crate, used for all functions that run a `libsais` algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibsaisError {
    /// Corresponds to the error code `-1` of the C library.
    InvalidInput,
    /// Corresponds to the error code `-2` of the C library.
    OutOfMemory,
    /// Corresponds to an unexpected error code other than `0` for success, `-1` and `-2`.
    /// You should hopefully never encounter this.
    UnknownError,
}

impl LibsaisError {
    fn from_return_code(return_code: i64) -> Self {
        match return_code {
            0 => panic!("Return code does not indicate an error"),
            -1 => Self::InvalidInput,
            -2 => Self::OutOfMemory,
            _ => Self::UnknownError,
        }
    }
}

impl Display for LibsaisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <LibsaisError as std::fmt::Debug>::fmt(self, f)
    }
}

impl std::error::Error for LibsaisError {}

// convenience methods for use inside this trait
pub(crate) trait IntoSaisResult {
    fn into_empty_sais_result(self) -> Result<(), LibsaisError>;

    fn into_primary_index_sais_result(self) -> Result<usize, LibsaisError>;
}

impl<O: OutputElement> IntoSaisResult for O {
    fn into_empty_sais_result(self) -> Result<(), LibsaisError> {
        let return_code: i64 = self.into();

        if return_code != 0 {
            Err(LibsaisError::from_return_code(return_code))
        } else {
            Ok(())
        }
    }

    fn into_primary_index_sais_result(self) -> Result<usize, LibsaisError> {
        let return_code: i64 = self.into();

        if return_code < 0 {
            Err(LibsaisError::from_return_code(return_code))
        } else {
            Ok(return_code as usize)
        }
    }
}

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
