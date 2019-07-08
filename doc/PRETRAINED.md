# Pretrained models

The following models are available **for academic use**. The following
pretrained models are available.

## Models

### German (Hamburg-style Dependencies)

The German models with Hamburg-style dependencies use
[STTS](https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.html)
tags and use the [dependency annotation
guidelines](http://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/pdf/foth_eine_umfassende_.pdf)
of the Hamburg Dependency Treebank.

### German (Universal Dependencies)

The German models with Hamburg-style dependencies use a combination of
[Universal](https://universaldependencies.org/u/pos/) and
[STTS](https://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.html)
part-of-speech tags. You can obtain pure Universal Tags or STTS tags
by splitting the part-of-speech tag on the dash (`-`). For instance, a
token with the tag `VERB-VVPP` has the Universal POS tag `VERB` and
the STTS tag `VVPP`.

The dependency relations follow the [Universal
Dependency](https://universaldependencies.org/u/dep/index.html)
guidelines.

### Dutch (Universal Dependencies)

The part-of-speech tagger produces tags of the format
`UniversalTag(-DutchFeature)*`. You can obtain pure Universal Tags by
splitting retreiving the initial part of the tag before the dash (`-`)
character. For instance, a token with the tag `VERB-inf` has the
Universal POS tag `VERB`.

The dependency relations follow the [Universal
Dependency](https://universaldependencies.org/u/dep/index.html)
guidelines.

## Docker images with models

We provide Docker images with sticker and models. The images can be
used to tag local CoNLL-X files:

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

### German (Hamburg-style Dependencies)

| Image                                  | Language | Task                           |
|:---------------------------------------|:---------|:-------------------------------|
| `danieldk/sticker:de-pos-20190615`     | German   | POS tagging                    |
| `danieldk/sticker:de-topo-20190615`    | German   | Topological field tagging      |
| `danieldk/sticker:de-deps-20190617`    | German   | Dependency parsing             |

### German (Universal Dependencies)

| Image                                  | Language | Task                           |
|:---------------------------------------|:---------|:-------------------------------|
| `danieldk/sticker:de-pos-ud-20190705`  | German   | POS tagging (universal tagset) |
| `danieldk/sticker:de-deps-ud-20190705` | German   | Dependency Parsing (UD)        |

### Dutch (Universal Dependencies)

| Image                                  | Language | Task                           |
|:---------------------------------------|:---------|:-------------------------------|
| `danieldk/sticker:nl-pos-ud-20190623`  | Dutch    | POS tagging (universal tagset) |
| `danieldk/sticker:nl-deps-ud-20190628` | Dutch    | Dependency parsing (UD)        |

## Nix packages with models

sticker models can be installed with Nix through [danieldk's Nix
repository](https://git.sr.ht/~danieldk/nix-packages). The following
packages are available.

### German (Hamburg-style Dependencies)

| Attribute                  | Language | Task                           |
|:---------------------------|:---------|:-------------------------------|
| `stickerModels.de-pos`     | German   | POS tagging                    |
| `stickerModels.de-topo`    | German   | Topological field tagging      |
| `stickerModels.de-deps`    | German   | Dependency parsing             |

### German (Universal Dependencies)

| Attribute                  | Language | Task                           |
|:---------------------------|:---------|:-------------------------------|
| `stickerModels.de-pos-ud`  | German   | POS tagging (universal tagset) |
| `stickerModels.de-deps-ud` | German   | Dependency parsing (UD)        |

### Dutch (Universal Dependencies)

| Attribute                  | Language | Task                           |
|:---------------------------|:---------|:-------------------------------|
| `stickerModels.nl-pos-ud`  | Dutch    | POS tagging (universal tagset) |
| `stickerModels.nl-deps-ud` | Dutch    | Dependency parsing (UD)        |

### Usage

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
