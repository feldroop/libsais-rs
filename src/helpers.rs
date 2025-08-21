use crate::type_model::{InputElement, OutputElement};

/// TODO doc
pub fn concatenate_strings<'a>(iter: impl IntoIterator<Item = &'a [u8]>) -> Vec<u8> {
    let strings: Vec<_> = iter.into_iter().collect();
    let needed_capacity = strings.iter().map(|&s| s.len()).sum::<usize>() + strings.len();
    let mut concatenated_string = Vec::with_capacity(needed_capacity);

    for s in strings {
        concatenated_string.extend_from_slice(s);
        concatenated_string.push(0)
    }

    concatenated_string
}

/// Computes the maximum value of the text and guarantees that all value are in the range
/// [0, max_value]. Therefore the alphabet size returned is max_value + 1.
/// Therefore the maximum value also has to be smaller than the maximum allowed value of `O`.
pub(crate) fn compute_and_validate_alphabet_size<I: InputElement, O: OutputElement>(
    text: &[I],
) -> Result<O, &'static str> {
    let mut min = I::ZERO;
    let mut max = I::ZERO;

    for c in text {
        min = min.min(*c);
        max = max.max(*c);
    }

    if min < I::ZERO {
        Err("Text cannot contain negative chars")
    } else {
        let found_max: i64 = max.into();
        let max_allowed: i64 = O::MAX.into();

        if found_max == max_allowed {
            Err("Text cannot contain the maximum value as a character")
        } else {
            Ok(O::try_from(found_max as usize + 1).unwrap())
        }
    }
}

pub fn is_suffix_array<I: InputElement, O: OutputElement>(
    text: &[I],
    maybe_suffix_array: &[O],
) -> bool {
    if text.is_empty() && maybe_suffix_array.is_empty() {
        return true;
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

    for indices in maybe_suffix_array.windows(2) {
        let previous = indices[0].into() as usize;
        let current = indices[1].into() as usize;

        // for the generalized suffix array, the zero char borders can be in a different order than
        // they would be in the normal suffix array
        if concatenated_text[previous] == I::ZERO && concatenated_text[current] == I::ZERO {
            continue;
        }

        if &concatenated_text[previous..] > &concatenated_text[current..] {
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

        if is_generalized_suffix_array {
            if longest_common_prefix_gsa(&text[first..], &text[second..]) != lcp_value as usize {
                return false;
            }
        } else {
            if longest_common_prefix(&text[first..], &text[second..]) != lcp_value as usize {
                return false;
            }
        }
    }

    true
}

fn longest_common_prefix<I: InputElement>(t1: &[I], t2: &[I]) -> usize {
    let mut lcp = 0;

    for (c1, c2) in std::iter::zip(t1, t2) {
        if c1 != c2 {
            break;
        }

        lcp += 1;
    }

    lcp
}

fn longest_common_prefix_gsa<I: InputElement>(t1: &[I], t2: &[I]) -> usize {
    let mut lcp = 0;

    for (c1, c2) in std::iter::zip(t1, t2) {
        if c1 != c2 || *c1 == I::ZERO || *c2 == I::ZERO {
            break;
        }

        lcp += 1;
    }

    lcp
}
