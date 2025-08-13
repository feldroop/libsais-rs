/*!
Raw bindings to the [`libsais`](https://github.com/IlyaGrebnov/libsais) C library by Ilya Grebnov.

The `libsais` library provides fast linear-time construction of suffix array (SA),
generalized suffix array (GSA), longest common prefix (LCP) array, permuted LCP (PLCP) array,
Burrows-Wheeler transform (BWT) and inverse BWT, based on the induced sorting algorithm
(with optional OpenMP support for multi-core parallel construction).

The OpenMP support is guarded by the `openmp` feature of this crate, which is enabled by default.
*/

#![allow(non_camel_case_types)]

extern crate openmp_sys;

/// Version of the library for 8-bit inputs smaller than 2GB (2147483648 bytes).
/// Also allows integer array inputs in some instances.
pub mod libsais;

/// Extension of the library for inputs larger or equal to 2GB (2147483648 bytes).
pub mod libsais64;

/// Independent version of `libsais` for 16-bit inputs.
pub mod libsais16;

/// Independent version of `libsais64` for 16-bit inputs.
pub mod libsais16x64;
