use libsais::construction::AuxIndicesSamplingRate;
use libsais::context::UnBwtContext;
use libsais::{BwtConstruction, ExtraSpace};
mod common;

use common::setup_basic_example;

#[test]
fn unbwt() {
    let (text, _, mut frequency_table, mut ctx) = setup_basic_example();

    let mut construction = BwtConstruction::for_text(text)
        .in_owned_buffer()
        .with_owned_temporary_array_buffer32(ExtraSpace::Recommended)
        .single_threaded()
        .with_context(&mut ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let bwt = construction
        .run()
        .expect("libsais bwt should run without errors");

    let mut unbwt_ctx = UnBwtContext::new_single_threaded();

    let mut unbwt = bwt
        .unbwt()
        .with_owned_temporary_array_buffer32()
        .single_threaded()
        .with_context(&mut unbwt_ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        unbwt = unbwt.with_frequency_table(&mut frequency_table);
    }

    let recovered_text = unbwt
        .run()
        .expect("libsais unbwt should run without errors");

    assert_eq!(recovered_text.as_slice(), text);
}

#[test]
fn unbwt_in_place_with_aux() {
    let (text, _, mut frequency_table, mut ctx) = setup_basic_example();
    let mut text_then_bwt_then_text = text.to_vec();

    let mut construction = BwtConstruction::replace_text(&mut text_then_bwt_then_text)
        .with_owned_temporary_array_buffer32(ExtraSpace::Recommended)
        .single_threaded()
        .with_context(&mut ctx)
        .with_aux_indices(AuxIndicesSamplingRate::from(2));

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let bwt = construction
        .run()
        .expect("libsais bwt should run without errors");

    let mut unbwt_ctx = UnBwtContext::new_single_threaded();

    let mut unbwt = bwt
        .unbwt()
        .replace_bwt()
        .single_threaded()
        .with_context(&mut unbwt_ctx);

    // SAFETY: the frequency table defined in the example is valid
    unsafe {
        unbwt = unbwt.with_frequency_table(&mut frequency_table);
    }

    let _ = unbwt
        .run()
        .expect("libsais unbwt should run without errors");

    assert_eq!(&text_then_bwt_then_text, text);
}
