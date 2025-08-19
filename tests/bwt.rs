use libsais::{
    BwtConstruction, ExtraSpace, SuffixArrayConstruction, construction::AuxIndicesSamplingRate,
    helpers,
};

mod common;

use common::setup_basic_example;

#[test]
fn libsais_bwt() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut config = BwtConstruction::for_text(text)
        .in_owned_buffer()
        .with_owned_temporary_suffix_array_buffer(ExtraSpace::Fixed { value: extra_space })
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res = config
        .construct()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error")
        .into_vec();

    assert_eq!(res.bwt_primary_index().unwrap(), 2);

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        &suffix_array,
        res.bwt()
    ));
}

#[test]
fn libsais_bwt_aux() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut config = BwtConstruction::for_text(text)
        .in_owned_buffer()
        .with_owned_temporary_suffix_array_buffer(ExtraSpace::Fixed { value: extra_space })
        .with_aux_indices(AuxIndicesSamplingRate::from(2))
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res = config
        .construct_with_aux_indices()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error")
        .into_vec();

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        &suffix_array,
        res.bwt()
    ));

    assert!(helpers::is_libsais_aux_indices(
        res.aux_indices(),
        &suffix_array,
        2
    ));
}

#[test]
fn libsais_bwt_in_text() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
    let mut text_and_later_bwt = text.to_owned();

    let mut config = BwtConstruction::replace_text(&mut text_and_later_bwt)
        .with_owned_temporary_suffix_array_buffer(ExtraSpace::Fixed { value: extra_space })
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res = config
        .construct_in_borrowed_buffer()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error")
        .into_vec();

    assert_eq!(res.unwrap(), 2);

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        &suffix_array,
        &text_and_later_bwt
    ));
}

#[test]
fn libsais_bwt_with_aux_in_text() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();
    let mut text_and_later_bwt = text.to_owned();

    let mut config = BwtConstruction::replace_text(&mut text_and_later_bwt)
        .with_owned_temporary_suffix_array_buffer(ExtraSpace::Fixed { value: extra_space })
        .with_aux_indices(AuxIndicesSamplingRate::from(2))
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let aux_indices = config
        .construct_with_aux_indices_in_borrowed_and_owned_buffers()
        .expect("libsais should run without an error");

    let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error")
        .into_vec();

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        &suffix_array,
        &text_and_later_bwt
    ));

    assert!(helpers::is_libsais_aux_indices(
        aux_indices.as_slice(),
        &suffix_array,
        2
    ));
}
