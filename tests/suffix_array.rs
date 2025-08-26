#[allow(unused)]
use libsais::{SuffixArrayConstruction, ThreadCount, suffix_array::ExtraSpace};
use libsais::{
    context::{Context, UnBwtContext},
    suffix_array::{AlphabetSize, SuffixArrayWithText},
};

mod common;

use common::*;

#[test]
fn empty_text() {
    let res: SuffixArrayWithText<u8, _, _> = SuffixArrayConstruction::for_text(&[])
        .in_owned_buffer32()
        .single_threaded()
        .run()
        .expect("libsais should run without an error");

    assert!(res.suffix_array().is_empty());

    assert!(is_suffix_array(res.text(), res.suffix_array()));
}

#[test]
fn libsais_basic() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .single_threaded()
        .with_context(&mut ctx)
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space });

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    assert!(is_suffix_array(text, res.suffix_array()));
}

#[test]
fn libsais_generalized_suffix_array() {
    let (text, extra_space, mut frequency_table, mut ctx) =
        setup_generalized_suffix_array_example();

    let mut construction = SuffixArrayConstruction::for_text(&text)
        .in_owned_buffer()
        .single_threaded()
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

    assert!(is_generalized_suffix_array(&text, res.suffix_array()));
}

#[test]
fn libsais_with_output_buffer() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
    let buffer_size = text.len() + extra_space;
    let mut suffix_array_buffer = vec![0; buffer_size];

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_borrowed_buffer(&mut suffix_array_buffer)
        .single_threaded()
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let _ = construction
        .run()
        .expect("libsais should run without an error");

    assert!(is_suffix_array(text, &suffix_array_buffer[..text.len()]));
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

    assert!(is_suffix_array(text, res.suffix_array()));
}

#[cfg(feature = "openmp")]
#[test]
fn libsais_omp_ctx() {
    use libsais::context::Context;

    let (text, extra_space, mut frequency_table, _) = setup_basic_example();
    let mut ctx = Context::new_multi_threaded(ThreadCount::openmp_default());

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .multi_threaded(ThreadCount::openmp_default())
        .with_context(&mut ctx)
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space });

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    assert!(is_suffix_array(text, res.suffix_array()));
}

#[test]
fn libsais_16input_extra_space_fixed() {
    let text = [3u16, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let res = SuffixArrayConstruction::for_text(&text)
        .in_owned_buffer32()
        .single_threaded()
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: 200 })
        .run()
        .expect("libsais should run without an error");

    assert!(is_suffix_array(&text, res.suffix_array()));
}

#[test]
fn libsais_32input_no_extra_space() {
    let mut text = [3i32, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let res = SuffixArrayConstruction::for_text_mut(&mut text)
        .in_owned_buffer32()
        .single_threaded()
        .with_extra_space_in_buffer(ExtraSpace::None)
        .run()
        .expect("libsais should run without an error");

    assert!(is_suffix_array(res.text(), res.suffix_array()));
}

#[test]
fn libsais_64input_alphabet_size() {
    let mut text = [3i64, 5, 2, 65, 1, 3, 2, 51, 2, 34, 51];

    let mut construction = SuffixArrayConstruction::for_text_mut(&mut text)
        .in_owned_buffer64()
        .single_threaded();

    // SAFETY: the alphabet size is correct and there are no negative values in the example
    unsafe {
        construction = construction.with_alphabet_size(AlphabetSize::new(66));
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    assert!(is_suffix_array(res.text(), res.suffix_array()));
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

    assert!(is_suffix_array(res.text(), res.suffix_array()));
}

#[test]
fn create_and_drop_all_contexts() {
    let c1 = Context::<u8, i32, _>::new_single_threaded();
    let c2 = Context::<u16, i32, _>::new_single_threaded();
    let c3 = Context::<u16, i32, _>::new_multi_threaded(ThreadCount::fixed(2));
    let c4 = Context::<u16, i32, _>::new_multi_threaded(ThreadCount::fixed(2));

    let uc1 = UnBwtContext::<u8, i32, _>::new_single_threaded();
    let uc2 = UnBwtContext::<u16, i32, _>::new_single_threaded();
    let uc3 = UnBwtContext::<u16, i32, _>::new_multi_threaded(ThreadCount::fixed(2));
    let uc4 = UnBwtContext::<u16, i32, _>::new_multi_threaded(ThreadCount::fixed(2));

    std::mem::drop(c1);
    std::mem::drop(c2);
    std::mem::drop(c3);
    std::mem::drop(c4);

    std::mem::drop(uc1);
    std::mem::drop(uc2);
    std::mem::drop(uc3);
    std::mem::drop(uc4);
}

#[test]
fn readme() {
    let text = b"barnabasbrabblesaboutbananas";
    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .multi_threaded(ThreadCount::openmp_default())
        .run()
        .expect("The example in the README should really work")
        .into_vec();

    assert!(is_suffix_array(text, &suffix_array));
}
