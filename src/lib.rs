pub mod bwt;
pub mod context;
pub mod error;
pub mod lcp;
pub mod plcp;
pub mod suffix_array;
pub mod type_model;
pub mod unbwt;

mod owned_or_borrowed;

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

//      put type_model things in the right places
//      derives of public types
//      small benchmarks
//      setup CI and such
//      good docs and examples, README (figure out good pub exports)
//      release-plz good release (also libsais-sys update)

//      wait for answers on ilya grebnov questions
//      figure out whether to use ParallelismUndebiced or no Parallelism at all
//      fix context drop/undecided issue in drop (also TODOs in safe comment)
//      make context sound via forcing parallelism decision before supplying context (probably for all contructions)
//      when context is there, could get rid of parallelism marker
//      context in into other marker type

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
