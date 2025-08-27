use libsais::{SuffixArrayConstruction, context::Context, suffix_array::ExtraSpace};

pub fn concatenate_texts<'a>(iter: impl IntoIterator<Item = &'a [u8]>) -> Vec<u8> {
    let texts: Vec<_> = iter.into_iter().collect();
    let needed_capacity = texts.iter().map(|t| t.len()).sum::<usize>() + texts.len();

    let mut concatenated_text = Vec::with_capacity(needed_capacity);

    for s in texts.into_iter() {
        concatenated_text.extend_from_slice(s);
        concatenated_text.push(0)
    }

    concatenated_text
}

fn main() {
    let texts = [
        b"bababababa".as_slice(),
        b"lalalalala",
        b"mamamamama",
        b"papapapapa",
    ];

    // To create a generalized suffix array (GSA), we need to first concatenate all of the indiviual texts
    // into one text, separated by the 0 value (not '0' in ASCII). The individual text should not contain this value.
    // The last character of the concatenated text also has to be 0.

    // it would also be possible to simply pass this text to the library with using the GSA mode.
    // The result would be a very similar suffix array. However, the tiebreaking behavior between the sentinels
    // (the zero separators) is only defined for the GSA mode. Without it, the tie-breaking behavior is
    // unpredicatable. For mroe information about this, consult the API documentation.
    let concatenated_text = concatenate_texts(texts);

    // The context allowes libsais to reuse memory allocations across invocations of the functions.
    // This is a small optimization that is only relevant when calling functions repeatedly
    // on smaller input texts (len < 64K).
    let mut context = Context::new_single_threaded();

    // The extra space in buffer let's libsais use additional memory. This might help with the performance
    // The default is to use the recommended extra space by the author of libsais.
    // The additional space is truncated before returning the suffix array.
    let mut construction = SuffixArrayConstruction::for_text(&concatenated_text)
        .in_owned_buffer32()
        .single_threaded()
        .generalized_suffix_array()
        .with_context(&mut context)
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: 25 });

    // The frequency table provides libsais with information about the ditribution of characters in the text
    // This is a completely optional parameter and might sometimes improve performance a little bit.
    let mut frequency_table = [0; 256];
    frequency_table[0] = 3;
    frequency_table[b'a' as usize] = 20;
    frequency_table[b'b' as usize] = 5;
    frequency_table[b'l' as usize] = 5;
    frequency_table[b'm' as usize] = 5;
    frequency_table[b'p' as usize] = 5;

    // SAFETY: the frequency table for this example is correct
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let generalized_suffix_array = construction.run().unwrap();

    assert!(generalized_suffix_array.is_generalized_suffix_array());
    println!(
        "Generalized suffix array: {:?}",
        generalized_suffix_array.suffix_array()
    );
}
