use std::env;

fn main() {
    let mut cc_build = cc::Build::new();

    cc_build
        .include("libsais/include")
        .file("libsais/src/libsais.c")
        .file("libsais/src/libsais16.c")
        .file("libsais/src/libsais16x64.c")
        .file("libsais/src/libsais64.c");

    if cfg!(feature = "openmp") {
        cc_build.flag("-DLIBSAIS_OPENMP");

        // this environment variable is defined by openms-sys
        env::var("DEP_OPENMP_FLAG")
            .unwrap()
            .split(" ")
            .for_each(|f| {
                cc_build.flag(f);
            });
    }

    cc_build.compile("sais");

    // The openmp-sys crate automatically tells Cargo to link lib(g)omp as appropriate.
    // However, some linkers are picky about the order in which libraries are specified,
    // and the automatic trick is not enough.
    if let Some(link) = env::var_os("DEP_OPENMP_CARGO_LINK_INSTRUCTIONS") {
        for i in env::split_paths(&link) {
            println!("cargo:{}", i.display());
        }
    }
}
