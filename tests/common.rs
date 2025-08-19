use libsais::{context::SingleThreaded8InputSaisContext, helpers};

pub fn setup_basic_example() -> (
    &'static [u8; 11],
    usize,
    [i32; 256],
    SingleThreaded8InputSaisContext,
) {
    let text = b"abababcabba";
    let extra_space = 10;
    let mut frequency_table = [0; 256];
    frequency_table[b'a' as usize] = 5;
    frequency_table[b'b' as usize] = 5;
    frequency_table[b'c' as usize] = 1;
    let ctx = SingleThreaded8InputSaisContext::new();

    (text, extra_space, frequency_table, ctx)
}

#[allow(dead_code)]
pub fn setup_generalized_suffix_array_example()
-> (Vec<u8>, usize, [i32; 256], SingleThreaded8InputSaisContext) {
    let text = helpers::concatenate_strings([b"abababcabba".as_slice(), b"babaabccbac"]);
    let extra_space = 20;
    let mut frequency_table = [0; 256];
    frequency_table[b'a' as usize] = 9;
    frequency_table[b'b' as usize] = 9;
    frequency_table[b'c' as usize] = 4;
    let ctx = SingleThreaded8InputSaisContext::new();

    (text, extra_space, frequency_table, ctx)
}
