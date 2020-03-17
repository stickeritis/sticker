# Pretrained models

## Models

### German (Universal Dependencies)

The German models use a combination of
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
$ docker run -i --rm danieldk/sticker:de-pos-ud-20190705 \
  /bin/sticker-tag-de-pos-ud < corpus.conll > tagged.conll
~~~

Or you can run the sticker server and expose it on a
local port:

~~~bash
$ docker run -it --rm -p 4000:4000 danieldk/sticker:de-pos-ud-20190705 \
  /bin/sticker-server-de-pos-ud 0.0.0.0:4000
~~~

### German (Universal Dependencies)

| Image                                        | Language | Task                           |
|:---------------------------------------------|:---------|:-------------------------------|
| `danieldk/sticker:de-pos-ud-20190705`        | German   | POS tagging (universal tagset) |
| `danieldk/sticker:de-deps-ud-large-20190926` | German   | Dependency Parsing (UD)        |
| `danieldk/sticker:de-ner-ud-small-20190928`  | German   | Named entity recognition       |
| `danieldk/sticker:de-topo-ud-small-20191002` | German   | Topological field prediction   |

### Dutch (Universal Dependencies)

| Image                                        | Language | Task                           |
|:---------------------------------------------|:---------|:-------------------------------|
| `danieldk/sticker:nl-pos-ud-20190623`        | Dutch    | POS tagging (universal tagset) |
| `danieldk/sticker:nl-deps-ud-large-20190929` | Dutch    | Dependency parsing (UD)        |
| `danieldk/sticker:nl-ner-ud-small-20191003`  | Dutch    | Named entity recognition       |

## Nix packages with models

sticker models can be installed with Nix through the [sticker Nix
package set](https://github.com/stickeritis/nix-packages). The following
packages are available.

### German (Universal Dependencies)

| Attribute                                 | Language | Task                           |
|:------------------------------------------|:---------|:-------------------------------|
| `sticker_models.de-pos-ud.wrapper`        | German   | POS tagging (universal tagset) |
| `sticker_models.de-deps-ud-small.wrapper` | German   | Dependency parsing (UD)        |
| `sticker_models.de-deps-ud-large.wrapper` | German   | Dependency parsing (UD)        |
| `sticker_models.de-ner-ud-small.wrapper`  | German   | Named entity recognition       |
| `sticker_models.de-topo-ud-small.wrapper` | German   | Topological field prediction   |

### Dutch (Universal Dependencies)

| Attribute                                | Language | Task                           |
|:-----------------------------------------|:---------|:-------------------------------|
| `stickerModels.nl-pos-ud.wrapper`        | Dutch    | POS tagging (universal tagset) |
| `stickerModels.nl-deps-ud-small.wrapper` | Dutch    | Dependency parsing (UD)        |
| `stickerModels.nl-deps-ud-large.wrapper` | Dutch    | Dependency parsing (UD)        |
| `stickerModels.nl-ner-ud-small.wrapper`  | Dutch    | Named entity recognition       |

### Usage

If you are not very familiar with Nix, the easiest way to install a
model is to install it into your local user environment. For example:

~~~bash
$ nix-env -f https://github.com/stickeritis/nix-packages/master.tar.gz \
  -iA stickerModels.de-pos-ud.wrapper
~~~

The packages have wrappers of the form `sticker-{tag,server}-model` to
call sticker with the applicable configuration file. For example if
you have installed `stickerModels.de-pos-ud.wrapper`, you could tag a
CoNLL-X file using:

~~~bash
$ sticker-tag-de-pos-ud corpus.conll tagged.conll
~~~

You can remove a model again using the `-e` flag of `nix-env`:

~~~
$ nix-env -e sticker-de-pos-ud
~~~

We recommend advanced users to add the [package
set](https://github.com/stickeritis/nix-packages) and use `nix-shell` to
create emphemeral environments with the models or to manage model
installation with [Home
Manager](https://github.com/rycee/home-manager).
