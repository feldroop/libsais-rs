use crate::type_model::{InputElementDecided, OutputElementDecided};

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
pub(crate) fn compute_and_validate_alphabet_size<
    I: InputElementDecided,
    O: OutputElementDecided,
>(
    text: &[I],
) -> Result<O, &'static str> {
    let zero = I::try_from(0).unwrap();
    let mut min = zero;
    let mut max = zero;

    for c in text {
        min = min.min(*c);
        max = max.max(*c);
    }

    if min < zero {
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

pub fn is_suffix_array<I: InputElementDecided, O: OutputElementDecided>(
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

pub fn is_generalized_suffix_array<I: InputElementDecided, O: OutputElementDecided>(
    concatenated_text: &[I],
    maybe_suffix_array: &[O],
) -> bool {
    if concatenated_text.is_empty() && maybe_suffix_array.is_empty() {
        return true;
    }

    let zero = I::try_from(0).unwrap();

    for indices in maybe_suffix_array.windows(2) {
        let previous = indices[0].into() as usize;
        let current = indices[1].into() as usize;

        // for the generalized suffix array, the zero char borders can be in a different order than
        // they would be in the normal suffix array
        if concatenated_text[previous] == zero && concatenated_text[current] == zero {
            continue;
        }

        if &concatenated_text[previous..] > &concatenated_text[current..] {
            return false;
        }
    }

    true
}

pub fn is_libsais_bwt<I: InputElementDecided, O: OutputElementDecided>(
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

    let mut i = 1;

    for &bwt_char in maybe_bwt[1..].iter() {
        let suffix_array_entry: i64 = suffix_array[i].into();

        // this would be the sentinel in the bwt, which is not there (virtual sentinel)
        if suffix_array_entry == 0 {
            // i is deliberately not incremented
            continue;
        }

        let rotated_index = if suffix_array_entry == 0 {
            text.len() - 1
        } else {
            suffix_array_entry as usize - 1
        };

        println!("{suffix_array_entry} {rotated_index}");
        if bwt_char.into() != text[rotated_index].into() {
            return false;
        }

        i += 1;
    }

    true
}
