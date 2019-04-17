use std::io::{Read, Write};

use failure::Error;
use sticker::depparse::{DependencyEncoding, RelativePOS, RelativePosition};
use sticker::Numberer;

use serde_cbor;
use toml;

use super::Config;

pub trait TomlRead {
    fn from_toml_read<R>(read: R) -> Result<Config, Error>
    where
        R: Read;
}

impl TomlRead for Config {
    fn from_toml_read<R>(mut read: R) -> Result<Self, Error>
    where
        R: Read,
    {
        let mut data = String::new();
        read.read_to_string(&mut data)?;
        let config: Config = toml::from_str(&data)?;
        Ok(config)
    }
}

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
cbor_read!(Numberer<String>);

pub trait CborWrite {
    fn to_cbor_write<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write;
}

// impl<T> CborWrite for Numberer<T>
// where
//     T: Eq + Hash + Serialize + Deserialize,
// {
//     fn to_cbor_write<W>(&self, write: &mut W) -> Result<()>
//     where
//         W: Write,
//     {
//         let data = serde_cbor::to_vec(self)?;
//         write.write(&data)?;
//         Ok(())
//     }
// }

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
cbor_write!(Numberer<String>);
