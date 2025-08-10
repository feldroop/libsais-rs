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

#[cfg(test)]
mod tests {
    use super::*;

    fn is_suffix_array(text: &[u8], maybe_suffix_array: &[i32]) -> bool {
        if text.is_empty() && maybe_suffix_array.is_empty() {
            return true;
        }

        for indices in maybe_suffix_array.windows(2) {
            let previous = indices[0] as usize;
            let current = indices[1] as usize;

            if &text[previous..] > &text[current..] {
                return false;
            }
        }

        true
    }

    #[test]
    fn libsais_basic() {
        let text = b"abababcabba";
        let mut suffix_array = [0; 11];
        let res = unsafe {
            libsais::libsais(
                text.as_ptr(),
                suffix_array.as_mut_ptr(),
                text.len() as i32,
                0,
                std::ptr::null_mut(),
            )
        };

        assert_eq!(res, 0);
        assert!(is_suffix_array(text, &suffix_array))
    }

    #[cfg(feature = "openmp")]
    #[test]
    fn libsais_omp() {
        let text = b"abababcabba";
        let mut suffix_array = [0; 11];
        let res = unsafe {
            libsais::libsais_omp(
                text.as_ptr(),
                suffix_array.as_mut_ptr(),
                text.len() as i32,
                0,
                std::ptr::null_mut(),
                2,
            )
        };

        assert_eq!(res, 0);
        assert!(is_suffix_array(text, &suffix_array))
    }
}
