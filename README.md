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

## Prediction

### Local

Given an existing model configuration such as `postag.conf`, you can
use `sticker-tag` to annotate data in CoNLL-X format:

~~~shell
$ sticker-tag postag.conf input.conll output.conll
~~~

When the input and output are not specified, `sticker-tag` will read
from the *standard input* and write to the *standard output*.

### Client/server

sticker can also run as a simple server:

~~~shell
$ sticker-server postag.conf localhost:4000
~~~

This will load the model defined in `postag.conf` and then listen on a
socket on `localhost` port `4000`. You can then send data in CoNLL-X
format to this port and the sticker will return the annotated
data. The last chunk of data will only be written if the client shuts
down the writing end off their socker (see the `shutdown(2)` manual
page).

## Training

In order to train a model, a model configuration file is needed. This file
describes settings such as which embeddings should be used. Sample
configuration files for various tasks can be found in the [examples](examples/)
directory. Given a configuration file, the first step is to create a shapes
file.

~~~shell
$ sticker-prepare postag.conf train.conll postag.shapes
~~~

This file is used in the construction of the Tensorflow
graph. Depending on which type of model you want to train, you can use
one of the `sticker-write-{conv,rnn,transformer}-graph` scripts. RNNs
are typically a good place to start. You can then define an RNN graph
as follows:

~~~shell
$ sticker-write-rnn-graph postag.shapes postag.graph
~~~

This creates a graph with the default hyperparameters. To list the
possible hyperparameters, use the `--help` option of the graph writing
script. After the graph is created, update the `graph` setting in the
configuration file (here `postag.conf`) to use the generated graph.

Finally, you can then train the model parameters:

~~~shell
$ sticker-train postag.conf train.conll validation.conll 
~~~

The models are quite sensitive to the learning rate and may diverge
when the learning rate is to high. The default learning rate is *0.01*
and can be set using the `lr` option. For example, to use a learning
rate of *0.001*, use:

~~~shell
$ sticker-train --lr 0.001 postag.conf train.conll validation.conll 
~~~

The training procedure will output the best epoch. Update the
`parameters` setting in the configuration file to use that epoch.


## Issues

You can report bugs and feature requests in the [sticker issue
tracker](https://github.com/danieldk/sticker/issues).
