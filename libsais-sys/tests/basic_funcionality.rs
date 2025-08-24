use libsais_sys::libsais;

fn is_suffix_array(text: &[u8], maybe_suffix_array: &[i32]) -> bool {
    if text.is_empty() && maybe_suffix_array.is_empty() {
        return true;
    }

    for indices in maybe_suffix_array.windows(2) {
        let previous = indices[0] as usize;
        let current = indices[1] as usize;

        if text[previous..] > text[current..] {
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
