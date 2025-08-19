mod common;

use common::{setup_basic_example, setup_generalized_suffix_array_example};
use libsais::{ExtraSpace, SuffixArrayConstruction};

#[test]
fn plcp() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut config = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .with_context(&mut ctx)
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space });

    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res1 = config
        .construct()
        .expect("libsais should run without an error");

    let _ = res1
        .plcp_construction()
        .construct()
        .expect("libsais plcp should run without an error");
}

#[test]
fn plcp_gsa() {
    let (text, extra_space, mut frequency_table, mut ctx) =
        setup_generalized_suffix_array_example();

    let mut config = SuffixArrayConstruction::for_text(&text)
        .in_owned_buffer()
        .with_context(&mut ctx)
        .generalized_suffix_array()
        .with_extra_space_in_buffer(ExtraSpace::Fixed { value: extra_space });

    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res1 = config
        .construct()
        .expect("libsais should run without an error");

    let _ = res1
        .plcp_construction()
        .construct()
        .expect("libsais plcp should run without an error");
}
