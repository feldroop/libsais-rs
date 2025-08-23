use crate::type_model::OutputElement;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibsaisError {
    InvalidInput,
    OutOfMemory,
    UnknownError,
}

impl LibsaisError {
    fn from_return_code(return_code: i64) -> Self {
        match return_code {
            0 => panic!("Return code does not indicate an error"),
            -1 => Self::InvalidInput,
            -2 => Self::OutOfMemory,
            _ => Self::UnknownError,
        }
    }
}

pub(crate) trait IntoSaisResult {
    fn into_empty_sais_result(self) -> Result<(), LibsaisError>;

    fn into_primary_index_sais_result(self) -> Result<usize, LibsaisError>;
}

impl<O: OutputElement> IntoSaisResult for O {
    fn into_empty_sais_result(self) -> Result<(), LibsaisError> {
        let return_code: i64 = self.into();

        if return_code != 0 {
            Err(LibsaisError::from_return_code(return_code))
        } else {
            Ok(())
        }
    }

    fn into_primary_index_sais_result(self) -> Result<usize, LibsaisError> {
        let return_code: i64 = self.into();

        if return_code < 0 {
            Err(LibsaisError::from_return_code(return_code))
        } else {
            Ok(return_code as usize)
        }
    }
}
