# sticker

**sticker** is a sequence labeler using neural networks.

## Introduction

sticker is a sequence labeler that uses either recurrent neural
networks or dilated convolution networks. In principle, it can be
used to perform any sequence labeling task, but so far the focus
has been on:

* Part-of-speech tagging
* Topological field tagging
* Dependency parsing

## Precompiled binaries

Precompiled binaries are available for Linux and macOS through [GitHub
releases](https://github.com/danieldk/sticker/releases). The binaries
are distributed with a precompiled CPU version of Tensorflow.

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

sticker should then be installed in `~/.cargo/bin/sticker-{tag,train,server}`

## Issues

You can report bugs and feature requests in the [sticker issue
tracker](https://github.com/danieldk/sticker/issues).
