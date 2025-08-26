use libsais::{
    BwtConstruction,
    bwt::{AuxIndicesSamplingRate, BwtWithAuxIndices},
    type_state::{BorrowedBuffer, OwnedBuffer},
};

fn main() {
    let text = b"barnabasbabblesaboutbananas";
    let mut text_copy = Vec::from_iter(text.iter().copied());

    // in this example, we replace the text in text_copy by the bwt, and then recover the text

    // first, we construct the bwt and destructure the result
    let res = BwtConstruction::replace_text(&mut text_copy)
        .with_owned_temporary_array_buffer32()
        .single_threaded()
        .with_aux_indices(AuxIndicesSamplingRate::from(4))
        .run()
        .unwrap();

    // bwt now is a reference to text_copy
    let (bwt, aux_indices, sampling_rate) = res.into_parts();

    println!(
        "Burrows-Wheeler-Transform: {}",
        std::str::from_utf8(bwt).unwrap()
    );

    println!(
        "Burrows-Wheeler-Transform in text_copy buffer: {}",
        std::str::from_utf8(&text_copy).unwrap()
    );

    // ... now you can do anything with text_copy, like compress and decompress it
    // when you have changed it back to the original state, you can recover the result object like so:

    // it's a bit ugly, because it's not possible to infer the BufferMode typestate from the input

    // SAFETY: we know that the aux indices are correct for this bwt and this sampling rate
    let recovered_res: BwtWithAuxIndices<_, _, BorrowedBuffer, OwnedBuffer> = unsafe {
        BwtWithAuxIndices::from_parts(text_copy.as_mut_slice(), aux_indices, sampling_rate)
    };

    // now we can recover the text and replace the bwt
    let res_unbwt = recovered_res
        .unbwt()
        .replace_bwt()
        .single_threaded()
        .run()
        .unwrap();

    println!(
        "Recovered text: {}",
        std::str::from_utf8(res_unbwt.as_slice()).unwrap()
    );

    // and finally, we check whether the recovered text was actually
    assert_eq!(text, res_unbwt.as_slice());
    assert_eq!(text.as_slice(), &text_copy);
}
