/// TODO doc
pub fn concatenate_strings_for_generalized_suffix_array<'a>(
    iter: impl IntoIterator<Item = &'a [u8]>,
) -> Vec<u8> {
    let strings: Vec<_> = iter.into_iter().collect();
    let needed_capacity = strings.iter().map(|&s| s.len()).sum::<usize>() + strings.len();
    let mut concatenated_string = Vec::with_capacity(needed_capacity);

    for s in strings {
        concatenated_string.extend_from_slice(s);
        concatenated_string.push(0)
    }

    concatenated_string
}
