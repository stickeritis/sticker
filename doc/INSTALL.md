# Installing sticker

## Precompiled binaries

Precompiled binaries are available for Linux and macOS through [GitHub
releases](https://github.com/danieldk/sticker/releases). The binaries
are distributed with a precompiled CPU version of Tensorflow.

## From source

Building sticker has the following requirements:

* A reasonably [modern Rust compiler](https://rustup.rs).
* Tensorflow built as a dynamic library (the Python module is **only** to construct/write the graph).

### macOS

Install the dependencies using Homebrew:

~~~bash
$ brew install rustup-init libtensorflow
# Install/configure the Rust toolchain.
$ rustup-init
~~~

Then compile and install sticker:

~~~bash
$ cd sticker
$ cargo install --path sticker-utils
~~~

sticker should then be installed in `~/.cargo/bin/sticker`
