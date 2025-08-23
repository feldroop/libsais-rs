#![allow(unused)]

use libsais::{
    context::Context,
    type_model::{InputElement, OutputElement, SingleThreaded},
};

pub fn setup_basic_example() -> (
    &'static [u8; 11],
    usize,
    [i32; 256],
    Context<u8, i32, SingleThreaded>,
) {
    let text = b"abababcabba";
    let extra_space = 10;
    let mut frequency_table = [0; 256];
    frequency_table[b'a' as usize] = 5;
    frequency_table[b'b' as usize] = 5;
    frequency_table[b'c' as usize] = 1;
    let ctx = Context::<_, _, SingleThreaded>::new_single_threaded();

    (text, extra_space, frequency_table, ctx)
}

#[allow(dead_code)]
pub fn setup_generalized_suffix_array_example()
-> (Vec<u8>, usize, [i32; 256], Context<u8, i32, SingleThreaded>) {
    let text = concatenate_strings([b"abababcabba".as_slice(), b"babaabccbac"]);
    let extra_space = 20;
    let mut frequency_table = [0; 256];
    frequency_table[b'a' as usize] = 9;
    frequency_table[b'b' as usize] = 9;
    frequency_table[b'c' as usize] = 4;
    let ctx = Context::new_single_threaded();

    (text, extra_space, frequency_table, ctx)
}

pub fn concatenate_strings<'a>(iter: impl IntoIterator<Item = &'a [u8]>) -> Vec<u8> {
    let mut concatenated_string = Vec::new();

    for s in iter.into_iter() {
        concatenated_string.extend_from_slice(s);
        concatenated_string.push(0)
    }

    concatenated_string
}

pub fn is_suffix_array<I: InputElement, O: OutputElement>(
    text: &[I],
    maybe_suffix_array: &[O],
) -> bool {
    if text.is_empty() && maybe_suffix_array.is_empty() {
        return true;
    }

    if text.len() != maybe_suffix_array.len() {
        return false;
    }

    for indices in maybe_suffix_array.windows(2) {
        let previous = indices[0].into() as usize;
        let current = indices[1].into() as usize;

        if &text[previous..] > &text[current..] {
            return false;
        }
    }

    true
}

pub fn is_generalized_suffix_array<I: InputElement, O: OutputElement>(
    concatenated_text: &[I],
    maybe_suffix_array: &[O],
) -> bool {
    if concatenated_text.is_empty() && maybe_suffix_array.is_empty() {
        return true;
    }

    if concatenated_text.len() != maybe_suffix_array.len() {
        return false;
    }

    let mut text_with_markers = Vec::new();

    let mut sentinel_weight = -1;

    for c in concatenated_text {
        if *c == I::ZERO {
            text_with_markers.push((*c, sentinel_weight));
            sentinel_weight -= 1;
        } else {
            text_with_markers.push((*c, 0));
        }
    }

    for indices in maybe_suffix_array.windows(2) {
        let previous = indices[0].into() as usize;
        let current = indices[1].into() as usize;

        let previous_full_suffix = &text_with_markers[previous..];
        let current_full_suffix = &text_with_markers[current..];

        let end_previous = previous_full_suffix
            .iter()
            .position(|&(c, _)| c == I::ZERO)
            .unwrap();

        let end_current = current_full_suffix
            .iter()
            .position(|&(c, _)| c == I::ZERO)
            .unwrap();

        let previous_suffix = if end_previous == previous_full_suffix.len() {
            previous_full_suffix
        } else {
            &previous_full_suffix[..end_previous]
        };
        let current_suffix = if end_current == current_full_suffix.len() {
            current_full_suffix
        } else {
            &current_full_suffix[..end_current]
        };

        if previous_suffix.is_empty() && current_suffix.is_empty() {
            continue;
        }

        if previous_suffix > current_suffix {
            return false;
        }
    }

    true
}

pub fn is_libsais_bwt<I: InputElement, O: OutputElement>(
    text: &[I],
    suffix_array: &[O],
    maybe_bwt: &[I],
) -> bool {
    // this is a bit complicated, because the libsais output does not include the sentinel.
    // neither its index in the suffix array, nor the char itself in the bwt
    if text.is_empty() && maybe_bwt.is_empty() {
        return true;
    }

    // first rotation in the burrows wheeler matrix always starts with the (virtual) sentinel.
    // therefore the first text char is at the top of the last column (the BWT)
    if maybe_bwt[0] != text[0] {
        return false;
    }

    // start with 1 because the index of the sentinel is not present at the beginning of the suffix array
    let mut i = 1;

    for &suffix_array_entry in suffix_array {
        let suffix_array_entry: i64 = suffix_array_entry.into();

        if suffix_array_entry == 0 {
            // this would be the sentinel in the bwt, which is not there (virtual sentinel)
            // i is delibaretly not incremented here
            continue;
        }

        let rotated_index = if suffix_array_entry == 0 {
            text.len() - 1
        } else {
            suffix_array_entry as usize - 1
        };

        let bwt_char = maybe_bwt[i];

        if bwt_char.into() != text[rotated_index].into() {
            return false;
        }

        i += 1;
    }

    true
}

pub fn is_libsais_aux_indices<O: OutputElement>(
    aux_indices: &[O],
    suffix_array: &[O],
    sampling_rate: usize,
) -> bool {
    // this is what the aux indices are defined as:
    // aux[i] == k => suffix_array[k - 1] = i * r

    for (i, &aux_index) in aux_indices.iter().enumerate() {
        let aux_index: i64 = aux_index.into();
        let suffix_array_entry: i64 = suffix_array[aux_index as usize - 1].into();

        if suffix_array_entry as usize != i * sampling_rate {
            return false;
        }
    }

    true
}

pub fn is_libsais_lcp<I: InputElement, O: OutputElement>(
    text: &[I],
    suffix_array: &[O],
    lcp: &[O],
    is_generalized_suffix_array: bool,
) -> bool {
    for (i, indices) in suffix_array.windows(2).enumerate() {
        let first = indices[0].into() as usize;
        let second = indices[1].into() as usize;

        let lcp_value = lcp[i + 1].into();

        if longest_common_prefix(&text[first..], &text[second..], is_generalized_suffix_array)
            != lcp_value as usize
        {
            return false;
        }
    }

    true
}

fn longest_common_prefix<I: InputElement>(
    t1: &[I],
    t2: &[I],
    is_generalized_suffix_array: bool,
) -> usize {
    let mut lcp = 0;

    for (c1, c2) in std::iter::zip(t1, t2) {
        if is_generalized_suffix_array && (*c1 == I::ZERO || *c2 == I::ZERO) {
            break;
        }

        if c1 != c2 {
            break;
        }

        lcp += 1;
    }

    lcp
}
