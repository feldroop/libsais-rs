use libsais::{
    BwtConstruction, SuffixArrayConstruction, bwt::AuxIndicesSamplingRate, suffix_array::ExtraSpace,
};

mod common;

use common::*;

#[test]
fn libsais_bwt_with_borrowed_temporary_array_buffer() {
    let (text, _, mut frequency_table, mut ctx) = setup_basic_example();

    let mut suffix_array_buffer = [0i32; 42];

    let mut construction = BwtConstruction::for_text(text)
        .in_owned_buffer()
        .with_borrowed_temporary_array_buffer(&mut suffix_array_buffer)
        .single_threaded()
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .single_threaded()
        .run()
        .expect("libsais should run without an error")
        .into_vec();

    assert_eq!(res.primary_index(), 2);

    assert!(is_libsais_bwt(text.as_slice(), &suffix_array, res.bwt()));
}

#[test]
fn libsais_bwt_aux() {
    let (text, _, mut frequency_table, mut ctx) = setup_basic_example();

    let mut construction = BwtConstruction::for_text(text)
        .in_owned_buffer()
        .with_owned_temporary_array_buffer()
        .single_threaded()
        .with_aux_indices(AuxIndicesSamplingRate::from(2))
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .single_threaded()
        .run()
        .expect("libsais should run without an error")
        .into_vec();

    assert!(is_libsais_bwt(text.as_slice(), &suffix_array, res.bwt()));

    assert!(is_libsais_aux_indices(res.aux_indices(), &suffix_array, 2));
}

#[test]
fn libsais_bwt_in_text() {
    let (text, _, mut frequency_table, mut ctx) = setup_basic_example();
    let mut text_and_later_bwt = text.to_owned();

    let mut construction = BwtConstruction::replace_text(&mut text_and_later_bwt)
        .with_owned_temporary_array_buffer()
        .single_threaded()
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .single_threaded()
        .run()
        .expect("libsais should run without an error")
        .into_vec();

    assert_eq!(res.primary_index(), 2);

    assert!(is_libsais_bwt(
        text.as_slice(),
        &suffix_array,
        &text_and_later_bwt
    ));
}

#[test]
fn libsais_bwt_with_aux_in_text() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
    let mut text_and_later_bwt = text.to_owned();

    let mut construction = BwtConstruction::replace_text(&mut text_and_later_bwt)
        .with_owned_temporary_array_buffer_and_extra_space(ExtraSpace::Fixed { value: extra_space })
        .single_threaded()
        .with_aux_indices(AuxIndicesSamplingRate::from(2))
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .single_threaded()
        .run()
        .expect("libsais should run without an error")
        .into_vec();

    assert!(is_libsais_bwt(text.as_slice(), &suffix_array, res.bwt()));

    assert!(is_libsais_aux_indices(res.aux_indices(), &suffix_array, 2));
}
