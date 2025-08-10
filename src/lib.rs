use std::ptr;

use libsais_sys::libsais;

pub use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

pub const LIBSAIS_MAXIMUM_TEXT_SIZE: usize = 2147483647;
pub const LIBSAIS16_MAXIMUM_TEXT_SIZE: usize = LIBSAIS_MAXIMUM_TEXT_SIZE;

pub struct SaisConfig<'a> {
    extra_space: usize,
    frequency_table: Option<&'a mut [i32; 256]>,
}

impl<'a> SaisConfig<'a> {
    pub fn new() -> Self {
        Self {
            extra_space: 0,
            frequency_table: None,
        }
    }

    pub fn with_extra_space(&mut self, extra_space: usize) -> &mut Self {
        self.extra_space = extra_space;
        self
    }

    /// By calling this function you are claiming that the frequency table is valid for the text
    /// for which this config is used later. Otherwise there is not guarantee for correct behavior
    /// of the C library. This table is used only for a single run, because it might be mutated by
    /// libsais.
    // TODO make nicer
    pub unsafe fn with_frequency_table(
        &mut self,
        frequency_table: &'a mut [i32; 256],
    ) -> &mut Self {
        self.frequency_table = Some(frequency_table);
        self
    }

    pub fn run(&mut self, text: &[u8]) -> Result<Vec<i32>, SaisError> {
        let buffer_len = text.len() + self.extra_space;
        let mut suffix_array_buffer = vec![0; buffer_len];

        let res = self.run_with_output_buffer(text, &mut suffix_array_buffer);

        suffix_array_buffer.truncate(text.len());

        res.map(|_| suffix_array_buffer)
    }

    pub fn run_with_output_buffer(
        &mut self,
        text: &[u8],
        suffix_array_buffer: &mut [i32],
    ) -> Result<(), SaisError> {
        assert!(text.len() < LIBSAIS_MAXIMUM_TEXT_SIZE);

        let expected_buffer_len = text.len() + self.extra_space;
        assert_eq!(suffix_array_buffer.len(), expected_buffer_len);

        let frequency_table_ptr = self
            .frequency_table
            .take()
            .map_or(ptr::null_mut(), |freq| freq.as_mut_ptr());

        // SAFETY:
        // text len is asserted to be in required range, which also makes the as i32 cast valid
        // suffix array buffer is asserted above to have the correct length
        // the library user claimed earlier that the frequency table is correct by calling an unsafe function
        let return_code = unsafe {
            libsais::libsais(
                text.as_ptr(),
                suffix_array_buffer.as_mut_ptr(),
                text.len() as i32,
                self.extra_space as i32,
                frequency_table_ptr,
            )
        };

        if return_code != 0 {
            Err(SaisError::from_return_code(return_code))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaisError {
    InvalidConfig,
    AlgorithmError,
    UnknownError,
}

impl SaisError {
    fn from_return_code(return_code: i32) -> Self {
        match return_code {
            0 => panic!("Return code does not indicate an error"),
            -1 => Self::InvalidConfig,
            -2 => Self::AlgorithmError,
            _ => Self::UnknownError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_suffix_array(text: &[u8], maybe_suffix_array: &[i32]) -> bool {
        if text.is_empty() && maybe_suffix_array.is_empty() {
            return true;
        }

        for indices in maybe_suffix_array.windows(2) {
            let previous = indices[0] as usize;
            let current = indices[1] as usize;

            if &text[previous..] > &text[current..] {
                return false;
            }
        }

        true
    }

    #[test]
    fn libsais_basic() {
        let text = b"abababcabba";
        let mut frequency_table = [0; 256];
        frequency_table[b'a' as usize] = 5;
        frequency_table[b'b' as usize] = 5;
        frequency_table[b'c' as usize] = 1;

        let mut config = SaisConfig::new();

        // SAFETY: the frequency table defined above is valid
        unsafe {
            config.with_frequency_table(&mut frequency_table);
        }

        let suffix_array = config
            .with_extra_space(5)
            .run(text)
            .expect("libsais you run without an error");

        assert!(is_suffix_array(text, &suffix_array));
    }
}
