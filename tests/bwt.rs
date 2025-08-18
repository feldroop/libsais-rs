use libsais::{
    BwtConstruction, ExtraSpace, SuffixArrayConstruction, construction::AuxIndicesSamplingRate,
    context::SingleThreaded8InputSaisContext, data_structures::SuffixArray, helpers,
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

#[test]
fn libsais_bwt() {
    let (text, extra_space, mut frequency_table, mut ctx) = setup_basic_example();

    let mut config = BwtConstruction::for_text(text)
        .in_owned_buffer()
        .with_owned_temporary_suffix_array_buffer(ExtraSpace::Fixed { value: extra_space })
        .with_context(&mut ctx);

    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res = config
        .construct()
        .expect("libsais should run without an error");

    let suffix_array: SuffixArray<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error");

    assert_eq!(res.bwt_primary_index().unwrap(), 2);

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        suffix_array.as_slice(),
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

    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res = config
        .construct_with_aux_indices()
        .expect("libsais should run without an error");

    let suffix_array: SuffixArray<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error");

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        suffix_array.as_slice(),
        res.bwt()
    ));

    assert!(helpers::is_libsais_aux_indices(
        res.aux_indices(),
        suffix_array.as_slice(),
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

    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let res = config
        .construct_in_borrowed_buffer()
        .expect("libsais should run without an error");

    let suffix_array: SuffixArray<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error");

    assert_eq!(res.unwrap(), 2);

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        suffix_array.as_slice(),
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

    unsafe {
        config = config.with_frequency_table(&mut frequency_table);
    }

    let aux_indices = config
        .construct_with_aux_indices_in_borrowed_and_owned_buffers()
        .expect("libsais should run without an error");

    let suffix_array: SuffixArray<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .construct()
        .expect("libsais should run without an error");

    assert!(helpers::is_libsais_bwt(
        text.as_slice(),
        suffix_array.as_slice(),
        &text_and_later_bwt
    ));

    assert!(helpers::is_libsais_aux_indices(
        aux_indices.as_slice(),
        suffix_array.as_slice(),
        2
    ));
}
