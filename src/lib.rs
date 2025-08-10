use libsais_sys::libsais::LIBSAIS_VERSION_STRING;

pub fn sais_version() {
    println!("Hello, world! libsais version: {LIBSAIS_VERSION_STRING}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version() {
        sais_version();
    }
}
