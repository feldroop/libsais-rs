# libsais-rs

[![Build Status](https://img.shields.io/github/actions/workflow/status/feldroop/libsais-rs/rust.yml?style=flat-square&logo=github)](https://github.com/feldroop/libsais-rs/actions)
[![Crates.io](https://img.shields.io/crates/v/libsais.svg?style=flat-square&logo=rust)](https://crates.io/crates/libsais)
[![Documentation](https://img.shields.io/docsrs/libsais?style=flat-square&logo=rust)](https://docs.rs/libsais)

An idiomatic and mostly safe API wrapper for the awesome and _very_ fast library [`libsais`] by Ilya Grebnov.

⚠️ **Warning:** this crate is not yet battle-tested, there might be bugs. The API is still subject to small changes. Any kind of feedback and suggestions via the issue tracker is highly appreciated!⚠️

## Functionality

This crate exposes the whole functionality of [`libsais`]. It might be useful to also check out the [documentation of the original library](https://github.com/IlyaGrebnov/libsais).

- Suffix array construction for `u8`/`u16`/`i32`/`i64` texts and `i32`/`i64` output arrays
- Generalized suffix arrays
- Longest common prefix arrays and permuted longest common prefix arrays
- Burrows-Wheeler-Transform and reversal
- Optional multithreading support via the `openmp` feature (enabled by default)

## Usage

This crate provides generic builder-like APIs for all of the features listed above. The following is a simple example of how to use this library to construct a suffix array in parallel:

```rust
use libsais::{SuffixArrayConstruction, ThreadCount};

let text = b"barnabasbabblesaboutbananas";
let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
    .in_owned_buffer()
    .multi_threaded(ThreadCount::openmp_default())
    .run()
    .expect("The example in the README should really work")
    .into_vec();
```

Please consult the [documentation] and the [examples](./examples) for more details on how to use this library.

## Performance and benchmarks

This library only adds a few safety checks, which should not impact performance in a relevant way. A notable exception is the suffix array construction for `i32` and `i64` inputs. Please consult the [documentation](https://docs.rs/libsais/latest/libsais/suffix_array/index.html#large-alphabets) for details.

Below are the results of a small benchmark of suffix array construction algorithms available on [crates.io](https://crates.io). The input was the human genome, truncated to 2 GB. Details about this benchmark can be found [here](https://github.com/feldroop/benchmark_crates_io_sacas).

The excellent performance of [`libsais`] is one of the main reasons why this API wrapper crate was created.

<img src="https://raw.githubusercontent.com/feldroop/benchmark_crates_io_sacas/refs/heads/master/plot/plot.svg" />

[`libsais`]: https://github.com/IlyaGrebnov/libsais
[documentation]: https://docs.rs/libsais/latest/libsais/
