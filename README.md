# libsais-rs

An idiomatic and mostly safe API wrapper for the awesome and _very_ fast library [`libsais`](https://github.com/IlyaGrebnov/libsais) by Ilya Grebnov.

⚠️ **Warning:** this crate is not yet battle-tested, there might be bugs. The API is still subject to small changes. ⚠️ 

## Features

This crate exposes the whole functionality of `libsais`. It might be useful to also check out the [documentation of the original library](https://github.com/IlyaGrebnov/libsais).

- Suffix array construction for `u8`/`u16`/`i32`/`i64` texts and `i32`/`i64` output arrays
- Generalized suffix arrays
- Longest common prefix arrays and permuted longest common prefix arrays
- Burrows-Wheeler transform and reversal
- Optional multithreading support

Please consult the [documentation](https://docs.rs/libsais/latest/libsais/) or [examples](./examples) for more details about how to use this library.

## Usage

This crate provies generic builder-like APIs for all of the features listed above. The following is a simple example of how to use this library to construct a suffix array in parallel:

```rust
use libsais::{SuffixArrayConstruction, ThreadCount};

let text = b"barnabasbrabblesaboutbananas";
let suffix_array: Vec<i32> = SuffixArrayConstruction::for_text(text)
    .in_owned_buffer()
    .multi_threaded(ThreadCount::openmp_default())
    .run()
    .expect("The example in the README should really work")
    .into_vec();
```

## Performance and benchmarks

This library only adds a few safety checks, which should not impact performance in a relevant way. A notable exception is the suffix array construction for `i32` and `i64` inputs. Please consult the [documentation](https://docs.rs/libsais/latest/libsais/) for details.

Below are the results of a small benchmark of suffix array construction algorithms available on [crates.io](https://crates.io). The input was the human genome, truncated to 2 GB. Details about this benchmark can be found [here](https://github.com/feldroop/benchmark_crates_io_sacas).

<img src="https://raw.githubusercontent.com/feldroop/benchmark_crates_io_sacas/refs/heads/master/plot/plot.svg" />
