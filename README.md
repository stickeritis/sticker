# sticker

**sticker** is a part-of-speech tagger using recurrent neural networks.

## Building

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

sticker should then be installed in ~/.cargo/bin/sticker-{tag,train,server}

## Issues and patches

You can report bugs and feature requests in the [sticker issue
tracker](https://todo.sr.ht/~danieldk/sticker). Send patches to:
[~danieldk/public-inbox@lists.sr.ht](mailto:~danieldk/public-inbox@lists.sr.ht)
