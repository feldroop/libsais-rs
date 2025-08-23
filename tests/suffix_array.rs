#[allow(unused)]
use libsais::{ExtraSpace, SuffixArrayConstruction, ThreadCount, helpers};

mod common;

use common::{setup_basic_example, setup_generalized_suffix_array_example};

#[test]
fn libsais_basic() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .with_context(&mut ctx)
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space });

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(text, res.suffix_array()));
}

#[test]
fn libsais_generalized_suffix_array() {
    let (text, extra_space, mut frequency_table, mut ctx) =
        setup_generalized_suffix_array_example();

    let mut construction = SuffixArrayConstruction::for_text(&text)
        .in_owned_buffer()
        .with_context(&mut ctx)
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space })
        .generalized_suffix_array();

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    assert!(helpers::is_generalized_suffix_array(
        &text,
        res.suffix_array()
    ));
}

#[test]
fn libsais_with_output_buffer() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
    let buffer_size = text.len() + extra_space;
    let mut suffix_array_buffer = vec![0; buffer_size];

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_borrowed_buffer(&mut suffix_array_buffer)
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let _ = construction
        .run()
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

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .multi_threaded(ThreadCount::openmp_default())
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space });

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(text, res.suffix_array()));
}

#[test]
fn libsais_16input_extra_space_fixed() {
    let text = [3u16, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let res = SuffixArrayConstruction::for_text(&text)
        .in_owned_buffer32()
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: 200 })
        .run()
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(&text, res.suffix_array()));
}

#[test]
fn libsais_32input_no_extra_space() {
    let mut text = [3i32, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let res = SuffixArrayConstruction::for_text_mut(&mut text)
        .in_owned_buffer32()
        .with_extra_space_in_buffer(ExtraSpace::None)
        .run()
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(res.text(), res.suffix_array()));
}

#[test]
fn libsais_64input_alphabet_size() {
    let mut text = [3i64, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let mut construction = SuffixArrayConstruction::for_text_mut(&mut text).in_owned_buffer64();

    // SAFETY: the alphabet size is correct and there are no negative values in the example
    unsafe {
        construction = construction.with_alphabet_size(66);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(res.text(), res.suffix_array()));
}

#[cfg(feature = "openmp")]
#[test]
fn libsais_64input_omp() {
    let mut text = [3i64, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let res = SuffixArrayConstruction::for_text_mut(&mut text)
        .in_owned_buffer64()
        .multi_threaded(ThreadCount::fixed(2))
        .run()
        .expect("libsais should run without an error");

    assert!(helpers::is_suffix_array(res.text(), res.suffix_array()));
}
