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
    ExtraSpace, LibsaisError, ThreadCount, bwt::BwtConstruction, lcp::LcpConstruction,
    plcp::PlcpConstruction, suffix_array::SuffixArrayConstruction, unbwt::UnBwt,
};

// functionality differences:
// libsais: full
// libsais16: (-2) no plcp_int_[omp] (int functions are repetitive anyway)
// libsais64, 16x64: (-2) no plcp_long_[omp], (-12) no*_ctx functions

// output structures: SA, SA+BWT, SA+BWT+AUX, GSA for multistring

// required extra config: aux -> sampling rate, alhpabet size for int array, unbwt primary index
// optional extra config: with context, unbwt context, omp, frequency table

// other queries: lcp from plcp and sa, plcp from sa/gsa and text, unbwt

// TODOs:

//      refactor owned_or_borrowed to be (more) typesafe
//      fix is libsais gsa for sentinel management

//      wait for answers on ilya grebnov questions
//      figure out whether to use ParallelismUndebiced or no Parallelism at all
//      fix context drop/undecided issue in drop (also TODOs in safe comment)
//      make context sound via forcing parallelism decision before supplying context (probably for all contructions)
//      when context is there, could get rid of parallelism marker
//      context in into other marker type

//      more tests ((empty text, bwt and sa), untested combinations, bwt with temp sa buf)
//      primary_index option? (depending on what happens with empty text)

//      put things in the right places (especially type_model) (with pub exports)
//      derives of public types
//      small benchmarks
//      setup CI and such
//      good docs and examples, README
//      test without openmp
//      release-plz good release (also libsais-sys)
