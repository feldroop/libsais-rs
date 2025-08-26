use std::collections::HashMap;

use libsais::{
    SuffixArrayConstruction,
    suffix_array::{AlphabetSize, ExtraSpace},
};

// small for demonstration purposes, but it should work with large values (up to the realm of billions)
const TEXT_LEN: usize = 10;

fn create_random_text() -> Vec<i64> {
    std::iter::repeat_with(rand::random)
        .filter(|&x| x >= 0)
        .take(TEXT_LEN)
        .collect()
}

// it would also be possible to do the dense encoding in the original text buffer
fn with_dense_alphabet(text: &[i64]) -> (Vec<i64>, usize) {
    let mut transformation_table: HashMap<_, _> = text.iter().map(|&c| (c, 0i64)).collect();

    let mut unique_characters: Vec<_> = transformation_table.keys().copied().collect();
    let alphabet_size = unique_characters.len();

    unique_characters.sort_unstable();

    for (i, c) in unique_characters.iter().enumerate() {
        *transformation_table.get_mut(c).unwrap() = i as i64;
    }

    unique_characters.resize(text.len(), 0);
    let mut dense_text = unique_characters;

    for (dense_c, original_c) in dense_text.iter_mut().zip(text) {
        *dense_c = transformation_table[original_c];
    }

    (dense_text, alphabet_size)
}

fn main() {
    // a text of random, positive i64 values. (negative values are NOT allowed)
    let initial_text = create_random_text();
    println!("Initial text: {initial_text:?}");

    // Constructing a suffix array of the initial text like this is a very bad idea, because the SAIS algorithm
    // implemented by libsais is sensitive to the alphabet size (referred to as k) of the input text.
    // It needs to hold in memory at least one array of size k during the execution of the algorithm.

    // All values of the input text need to be in the range [0, k). It follows that for a random text,
    // k might have to be as large as i64::MAX.

    // To solve this issue, we create a new "dense" text. To this end, we first calculate the
    // minimum necessary alphabet size k_min by counting the number of distinct values. Then, we map the values
    // of the original text to the range [0, k_min), preserving their relativ order. For example, the maximum
    // of the original text becomes k_min -1, and so on.
    let (mut dense_text, alphabet_size) = with_dense_alphabet(&initial_text);
    println!("Dense text: {dense_text:?}\nAlphabet size: {alphabet_size}");

    // For large alphabets, the algorithm needs mutable access to the input text.
    // If there is no error, the text will be returned to its original state.
    let mut construction = SuffixArrayConstruction::for_text_mut(&mut dense_text)
        .in_owned_buffer64()
        .single_threaded();

    // The default is to use a recommended extra space of a couple thousand bytes for large alphabets.
    // For this example, we deactivate this behavior (just for show).
    construction = construction.with_extra_space_in_buffer(ExtraSpace::None);

    // This is unsafe, because undefined behavior can happen if there are negative values in the text
    // or if there are values equal to are larger than the alphabet size in the text.
    // It is still recommended to always use this function, because otherwise this API wrapper library
    // will have to inject a linear scan of the text to determine the alphabet size and make sure that
    // no negative values exist.
    unsafe {
        construction = construction.with_alphabet_size(AlphabetSize::new(alphabet_size as i64));
    }

    // Finally, we can run it. The resulting suffix array is valid for dense_text AND initial_text!
    let res = construction.run().unwrap();
    println!("Suffix array: {:?}", res.suffix_array());
}
