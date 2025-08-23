use libsais::{SuffixArrayConstruction, suffix_array::ExtraSpace};

mod common;

use common::*;

#[test]
fn empty_text_plcp_lcp() {
    let text: [u8; 0] = [];

    let res = SuffixArrayConstruction::for_text(&text)
        .in_owned_buffer32()
        .run()
        .expect("libsais should run without an error")
        .plcp_construction()
        .run()
        .expect("libsais plcp should run without an error")
        .lcp_construction()
        .run()
        .expect("libsais lcp should run without an error");

    assert!(res.suffix_array().is_empty());
    assert!(res.lcp().is_empty());

    assert!(is_libsais_lcp(&text, res.suffix_array(), res.lcp(), false));
}

#[test]
fn plcp_lcp_in_buffers() {
    let (text, _, mut frequency_table, mut ctx) = setup_basic_example();

    let mut suffix_array_buffer = [0i32; 11];
    let mut plcp_buffer = [0i32; 11];
    let mut lcp_buffer = [0i32; 11];

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_borrowed_buffer(&mut suffix_array_buffer)
        .with_context(&mut ctx);

    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error")
        .plcp_construction()
        .in_borrowed_buffer(&mut plcp_buffer)
        .run()
        .expect("libsais plcp should run without an error")
        .lcp_construction()
        .in_borrowed_buffer(&mut lcp_buffer)
        .run()
        .expect("libsais lcp should run without an error");

    assert!(is_libsais_lcp(text, res.suffix_array(), res.lcp(), false));
}

#[test]
fn plcp_lcp_gsa() {
    let (text, extra_space, mut frequency_table, mut ctx) =
        setup_generalized_suffix_array_example();

    let mut construction = SuffixArrayConstruction::for_text(&text)
        .in_owned_buffer()
        .with_context(&mut ctx)
        .generalized_suffix_array()
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space });

    unsafe {
        construction = construction.with_frequency_table(&mut frequency_table);
    }

    let res = construction
        .run()
        .expect("libsais should run without an error")
        .plcp_construction()
        .run()
        .expect("libsais plcp should run without an error")
        .lcp_construction()
        .run()
        .expect("libsais lcp should run without an error");

    assert!(is_libsais_lcp(&text, res.suffix_array(), res.lcp(), true));
}
