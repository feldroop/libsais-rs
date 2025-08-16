use libsais::{
    BwtConstruction, ExtraSpace, SuffixArrayConstruction, ThreadCount,
    context::SingleThreaded8InputSaisContext, helpers,
};

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

    let mut config = SuffixArrayConstruction::single_threaded()
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

    println!("{suffix_array:?}");

    assert!(helpers::is_suffix_array(text, suffix_array.as_slice()));
}

#[test]
fn libsais_generalized_suffix_array() {
    let (text, extra_space, mut frequency_table, mut ctx) =
        setup_generalized_suffix_array_example();

    let mut config = SuffixArrayConstruction::single_threaded()
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

    assert!(helpers::is_generalized_suffix_array(
        &text,
        suffix_array.as_slice()
    ));
}

#[test]
fn libsais_with_output_buffer() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
    let buffer_size = text.len() + extra_space;
    let mut suffix_array_buffer = vec![0; buffer_size];

    let mut config = SuffixArrayConstruction::single_threaded()
        .input_8_bits()
        .output_32_bits()
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        config = config.frequency_table(&mut frequency_table);
    }

    let _ = config
        .run_in_output_buffer(text, &mut suffix_array_buffer)
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(
        text,
        &suffix_array_buffer[..text.len()]
    ));
}

#[cfg(feature = "openmp")]
#[test]
fn libsais_omp() {
    let (text, extra_space, mut frequency_table, _) = setup_basic_example();

    let mut config = SuffixArrayConstruction::multi_threaded()
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

    assert!(helpers::is_suffix_array(text, suffix_array.as_slice()));
}

#[test]
fn libsais_16input_extra_space_fixed() {
    let text = [3u16, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let suffix_array = SuffixArrayConstruction::single_threaded()
        .input_16_bits()
        .output_64_bits()
        .run(&text, ExtraSpace::Fixed { value: 200 })
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(&text, suffix_array.as_slice()));
}

#[test]
fn libsais_32input_extra_space_recommended() {
    let mut text = [3i32, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let suffix_array = SuffixArrayConstruction::single_threaded()
        .input_and_output_32_bits()
        .run_large_alphabet(&mut text, ExtraSpace::Recommended)
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(&text, suffix_array.as_slice()));
}

#[test]
fn libsais_64input_alphabet_size() {
    let mut text = [3i64, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let mut config = SuffixArrayConstruction::single_threaded().input_and_output_64_bits();

    // SAFETY: the alphabet size is correct and there are no negative values in the example
    unsafe {
        config = config.alphabet_size(66);
    }

    let suffix_array = config
        .run_large_alphabet(&mut text, ExtraSpace::Recommended)
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(&text, suffix_array.as_slice()));
}

#[cfg(feature = "openmp")]
#[test]
fn libsais_64input_omp() {
    let mut text = [3i64, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let suffix_array = SuffixArrayConstruction::multi_threaded()
        .input_and_output_64_bits()
        .num_threads(ThreadCount::fixed(2))
        .run_large_alphabet(&mut text, ExtraSpace::Recommended)
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(&text, suffix_array.as_slice()));
}

#[test]
fn libsais_bwt() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut config = BwtConstruction::single_threaded()
        .input_8_bits()
        .output_32_bits()
        .with_context(&mut ctx);

    unsafe {
        config = config.frequency_table(&mut frequency_table);
    }

    let res = config
        .run(text.as_slice(), ExtraSpace::Fixed { value: extra_space })
        .expect("libsais should run without an error");

    let suffix_array = SuffixArrayConstruction::single_threaded()
        .input_8_bits()
        .output_32_bits()
        .run(text.as_slice(), ExtraSpace::None)
        .expect("libsais should run without an error");

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        suffix_array.as_slice(),
        res.bwt()
    ));
}
