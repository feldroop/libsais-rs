use libsais::{ExtraSpace, Sais, ThreadCount, context::SingleThreaded8InputSaisContext, helpers};

fn is_suffix_array(text: &[u8], maybe_suffix_array: &[i32]) -> bool {
    if text.is_empty() && maybe_suffix_array.is_empty() {
        return true;
    }

    for indices in maybe_suffix_array.windows(2) {
        let previous = indices[0] as usize;
        let current = indices[1] as usize;

        if &text[previous..] > &text[current..] {
            return false;
        }
    }

    true
}

fn is_generalized_suffix_array(concatenated_text: &[u8], maybe_suffix_array: &[i32]) -> bool {
    if concatenated_text.is_empty() && maybe_suffix_array.is_empty() {
        return true;
    }

    for indices in maybe_suffix_array.windows(2) {
        let previous = indices[0] as usize;
        let current = indices[1] as usize;

        // for the generalized suffix array, the zero char borders can be in a different order than
        // they would be in the normal suffix array
        if concatenated_text[previous] == 0 && concatenated_text[current] == 0 {
            continue;
        }

        if &concatenated_text[previous..] > &concatenated_text[current..] {
            return false;
        }
    }

    true
}

fn setup_basic_example() -> (
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

fn setup_generalized_suffix_array_example()
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

#[test]
fn libsais_basic() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut config = Sais::single_threaded()
        .input_8_bits()
        .output_32_bits()
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined above is valid
    unsafe {
        config = config.frequency_table(&mut frequency_table);
    }

    let suffix_array = config
        .run(text, ExtraSpace::Fixed { value: extra_space })
        .expect("libsais should run without an error");

    assert!(is_suffix_array(text, &suffix_array));
}

#[test]
fn libsais_generalized_suffix_array() {
    let (text, extra_space, mut frequency_table, mut ctx) =
        setup_generalized_suffix_array_example();

    let mut config = Sais::single_threaded()
        .input_8_bits()
        .output_32_bits()
        .with_context(&mut ctx)
        .generalized_suffix_array();

    // SAFETY: the frequency table defined above is valid
    unsafe {
        config = config.frequency_table(&mut frequency_table);
    }

    let suffix_array = config
        .run(&text, ExtraSpace::Fixed { value: extra_space })
        .expect("libsais should run without an error");

    assert!(is_generalized_suffix_array(&text, &suffix_array));
}

#[test]
fn libsais_with_output_buffer() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
    let buffer_size = text.len() + extra_space;
    let mut suffix_array_buffer = vec![0; buffer_size];

    let mut config = Sais::single_threaded()
        .input_8_bits()
        .output_32_bits()
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        config = config.frequency_table(&mut frequency_table);
    }

    let _ = config
        .run_with_output_buffer(text, &mut suffix_array_buffer)
        .expect("libsais should run without an error");

    assert!(is_suffix_array(text, &suffix_array_buffer[..text.len()]));
}

#[cfg(feature = "openmp")]
#[test]
fn libsais_omp() {
    let (text, extra_space, mut frequency_table, _) = setup_basic_example();

    let mut config = Sais::multi_threaded()
        .input_8_bits()
        .output_32_bits()
        .num_threads(ThreadCount::openmp_default());

    // SAFETY: the frequency table defined above is valid
    unsafe {
        config = config.frequency_table(&mut frequency_table);
    }

    let suffix_array = config
        .run(text, ExtraSpace::Fixed { value: extra_space })
        .expect("libsais should run without an error");

    assert!(is_suffix_array(text, &suffix_array));
}
