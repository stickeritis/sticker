# Installing sticker

## Precompiled binaries

Precompiled binaries are available for Linux and macOS through [GitHub
releases](https://github.com/danieldk/sticker/releases). The binaries
are distributed with a precompiled CPU version of Tensorflow.

## Docker images with models

The following prebuilt-docker images with models are available **for
academic use**.

| Image                               | Language | Task                      |
|:------------------------------------|:---------|:--------------------------|
| `danieldk/sticker:de-pos-20190615`  | German   | POS tagging               |
| `danieldk/sticker:de-topo-20190615` | German   | Topological field tagging |
| `danieldk/sticker:de-deps-20190617` | German   | Dependency parsing        |

The images can be used to tag local CoNLL-X files:

~~~bash
$ docker run -i --rm danieldk/sticker:de-pos-20190615 \
  /bin/sticker-tag-de-pos < corpus.conll > tagged.conll
~~~

Or you can run the sticker server and expose it on a
local port:

~~~bash
$ docker run -it --rm -p 4000:4000 danieldk/sticker:de-pos-20190615 \
  /bin/sticker-server-de-pos 0.0.0.0:4000
~~~

## Nix packages with models

sticker models can be installed with Nix through [danieldk's Nix
repository](https://git.sr.ht/~danieldk/nix-packages). The following
packages are available.

| Attribute               | Language | Task                      |
|:------------------------|:---------|:--------------------------|
| `stickerModels.de-pos`  | German   | POS tagging               |
| `stickerModels.de-topo` | German   | Topological field tagging |
| `stickerModels.de-deps` | German   | Dependency parsing        |

If you are not very familiar with Nix, the easiest way to install a
model is to install it into your local user environment. For example:

~~~bash
$ nix-env -f https://git.sr.ht/~danieldk/nix-packages/archive/master.tar.gz \
  -iA stickerModels.de-pos
~~~

The packages have wrappers of the form `sticker-{tag,server}-model`
to call sticker with the applicable configuration file. For example
if you have installed `stickerModels.de-pos`, you could tag a CoNLL-X
file using:

~~~bash
$ sticker-server-de-pos corpus.conll tagged.conll
~~~

You can remove a model again using the `-e` flag of `nix-env`:

~~~
$ nix-env -e sticker-de-pos
~~~

We recommend advanced users to add the [package
set](https://git.sr.ht/~danieldk/nix-packages) and use `nix-shell` to
create emphemeral environments with the models or to manage model
installation with [Home
Manager](https://github.com/rycee/home-manager).

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

sticker should then be installed in `~/.cargo/bin/sticker-{tag,train,server}`
