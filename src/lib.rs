pub mod construction;
pub mod context;
pub mod data_structures;
pub mod helpers;
pub mod type_model;

/// The version of the C library libsais wrapped by this crate
pub use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

/// The maximum text size that this library can handle when using i32-based buffers
pub const LIBSAIS_I32_OUTPUT_MAXIMUM_SIZE: usize = 2147483647;

/// The maximum text size that this library can handle when using i64-based buffers
pub const LIBSAIS_I64_OUTPUT_MAXIMUM_SIZE: usize = 9223372036854775807;

#[doc(inline)]
pub use construction::{
    ExtraSpace, SaisError, ThreadCount, bwt::BwtConstruction, suffix_array::SuffixArrayConstruction,
};

// functionality differences:
// libsais: full
// libsais16: (-2) no plcp_int_[omp] (int functions are repetitive anyway)
// libsais64, 16x64: (-2) no plcp_long_[omp], (-12) no*_ctx functions

// output structures: SA, SA+BWT, SA+BWT+AUX, GSA for multistring

// required extra config: aux -> sampling rate, alhpabet size for int array, unbwt primary index
// optional extra config: with context, unbwt context, omp, frequency table

// other queries: lcp from plcp and sa, plcp from sa/gsa and text, unbwt

// BIG TODOs:
//      UNBWT/LCP

// SMALL TODOs:

//      UNBWT: add new context types
//      UNBWT/LCP: wire operations on the backend
//      UNBWT/LCP: implement Constructions
//      UNBWT/LCP: find a way to make secondary operations such as unbwt and plcp/lcp safe?

//      maybe actually go back to traits for builder code sharing
//      find a way to make extra_space in bwt nicer
//      add well-defined type decision setter for contructions that allow generic use

//      more tests
//      seal traits
//      clean up using macros?
