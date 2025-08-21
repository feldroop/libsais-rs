mod common;

use common::{setup_basic_example, setup_generalized_suffix_array_example};
use libsais::{ExtraSpace, SuffixArrayConstruction, helpers};

#[test]
fn plcp() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut construction = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .with_context(&mut ctx)
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

    assert!(helpers::is_libsais_lcp(
        text,
        res.suffix_array(),
        res.lcp(),
        false
    ));
}

#[test]
fn plcp_gsa() {
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

    assert!(helpers::is_libsais_lcp(
        &text,
        res.suffix_array(),
        res.lcp(),
        true
    ));
}
