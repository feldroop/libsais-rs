use libsais::SuffixArrayConstruction;

fn main() {
    let text = b"barnabasbabblesaboutbananas";

    // To obtain a suffix array and the lcp array using this library, we also need to compute a permuted lcp array.
    // You can read more about these data structures and conventions of this library in the APi documentation.

    // first, we create the suffix array
    let suffix_array = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer32()
        .single_threaded()
        .run()
        .unwrap();

    // then we can add the plcp to the mix
    let suffix_array_and_plcp = suffix_array
        .plcp_construction()
        .single_threaded()
        .run()
        .unwrap();

    // and finally, we can compute the lcp
    let suffix_array_and_plcp_and_lcp = suffix_array_and_plcp
        .lcp_construction()
        .single_threaded()
        .run()
        .unwrap();

    // we can destructure the returned object and drop the plcp and information about whether is this
    // a generalized suffix array (we know its not)
    let (suffix_array, lcp, _, _) = suffix_array_and_plcp_and_lcp.into_parts();

    println!("Suffix array: {suffix_array:?}");
    println!("Longest common prefix array: {lcp:?}");
}
