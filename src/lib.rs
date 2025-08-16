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
pub use construction::{BwtConstruction, ExtraSpace, SuffixArrayConstruction, ThreadCount}; //, suffix_array::SuffixArrayConstruction};

// functionality differences:
// libsais: full
// libsais16: (-2) no plcp_int_[omp] (int functions are repetitive anyway)
// libsais64, 16x64: (-2) no plcp_long_[omp], (-12) no*_ctx functions

// output structures: SA, SA+BWT, SA+BWT+AUX, GSA for multistring

// required extra config: aux -> sampling rate, alhpabet size for int array, unbwt primary index
// optional extra config: with context, unbwt context, omp, frequency table

// other queries: lcp from plcp and sa, plcp from sa/gsa and text, unbwt

// BIG TODOs:
//      bwt + aux
//      unbwt, plcp + lcp

// SMALL TODOs:
//      BWT/AUX: add way to replace text in place
//      learning: bwt does not output suffix array -> split builders (find way to reuse impls, also needed for unbwt, maybe lcp)

//      find a way to make arguments such as extra_space, aux_indices_sampling_rate nicer:
//          Buffer Mode : AllocateAndReturn, AllocateAndReturnWithExtraSpace, WithGivenBuffer, BWTInplace?

//      find a way to make secondary operations such as ubwt and lcp safe

//      more tests (bwt)
//      seal traits
//      clean up using macros?
