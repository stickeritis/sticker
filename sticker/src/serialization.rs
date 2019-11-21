//! Serialization of some crate-specific types.

use std::io::{Read, Write};

use crate::encoder::deprel::{DependencyEncoding, RelativePOS, RelativePosition};
use crate::encoder::lemma::EditTree;
use crate::Numberer;
use failure::Error;

use serde_cbor;

pub trait CborRead
where
    Self: Sized,
{
    fn from_cbor_read<R>(read: R) -> Result<Self, Error>
    where
        R: Read;
}

macro_rules! cbor_read {
    ($type: ty) => {
        impl CborRead for $type {
            fn from_cbor_read<R>(read: R) -> Result<Self, Error>
            where
                R: Read,
            {
                let labels = serde_cbor::from_reader(read)?;
                Ok(labels)
            }
        }
    };
}

cbor_read!(Numberer<DependencyEncoding<RelativePOS>>);
cbor_read!(Numberer<DependencyEncoding<RelativePosition>>);
cbor_read!(Numberer<EditTree>);
cbor_read!(Numberer<String>);

pub trait CborWrite {
    fn to_cbor_write<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write;
}

macro_rules! cbor_write {
    ($type: ty) => {
        impl CborWrite for $type {
            fn to_cbor_write<W>(&self, write: &mut W) -> Result<(), Error>
            where
                W: Write,
            {
                let data = serde_cbor::to_vec(self)?;
                write.write_all(&data)?;
                Ok(())
            }
        }
    };
}

cbor_write!(Numberer<DependencyEncoding<RelativePOS>>);
cbor_write!(Numberer<DependencyEncoding<RelativePosition>>);
cbor_write!(Numberer<EditTree>);
cbor_write!(Numberer<String>);
