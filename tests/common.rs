use libsais::{context::Context, helpers, type_model::SingleThreaded};

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
    let text = helpers::concatenate_strings([b"abababcabba".as_slice(), b"babaabccbac"]);
    let extra_space = 20;
    let mut frequency_table = [0; 256];
    frequency_table[b'a' as usize] = 9;
    frequency_table[b'b' as usize] = 9;
    frequency_table[b'c' as usize] = 4;
    let ctx = Context::new_single_threaded();

    (text, extra_space, frequency_table, ctx)
}
