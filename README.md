**Warning:** sticker is succeeded by
[SyntaxDot](https://github.com/tensordot/syntaxdot), which supports
*many* new features:

* Multi-task learning.
* Pretrained transformer models, suchs as BERT and XLM-R.
* Biaffine parsing in addition to parsing as sequence labeling.
* Lemmatization.

# sticker

**sticker** is a sequence labeler using neural networks.

## Introduction

sticker is a sequence labeler that uses either recurrent neural
networks, transformers, or dilated convolution networks. In principle,
it can be used to perform any sequence labeling task, but so far the
focus has been on:

* Part-of-speech tagging
* Topological field tagging
* Dependency parsing
* Named entity recognition

## Features

* Input representations:
  * [finalfusion](https://finalfusion.github.io/) embeddings with subword units
  * Bidirectional byte LSTMs
* Hidden representations:
  * Bidirectional recurrent neural networks (LSTM or GRU)
  * Transformers
  * Dillated convolutions
* Classification layers:
  * Softmax (best-N)
  * CRF
* Deployment:
  * Standalone binary that links against `libtensorflow`
  * Very liberal [license](LICENSE.md)
  * Docker containers with models

## Status

sticker is almost production-ready and we are preparing for release
1.0.0. Graphs and models crated with the current version **must** work
with sticker 1.x.y. There may still be breaking API or configuration
file changes until 1.0.0 is released.

## Where to go from here

* [Installing sticker](doc/INSTALL.md)
* [Pretrained models](doc/PRETRAINED.md)
* [Using sticker](doc/USAGE.md)
* [Training a sticker model](doc/TRAIN.md)

## References

sticker uses techniques from or was inspired by the following papers:

* [Finding Function in Form: Compositional Character Models for Open
  Vocabulary Word
  Representation](https://aclweb.org/anthology/papers/D/D15/D15-1176/). Wang
  Ling, Chris Dyer, Alan W Black, Isabel Trancoso, Ramón Fermandez,
  Silvio Amir, Luís Marujo, Tiago Luís, 2015, Proceedings of the 2015
  Conference on Empirical Methods in Natural Language Processing
* [Transition-based dependency parsing with topological
  fields](https://aclweb.org/anthology/papers/P/P16/P16-2001/). Daniël
  de Kok, Erhard Hinrichs, 2016, Proceedings of the 54th Annual
  Meeting of the Association for Computational Linguistics
* [Viable Dependency Parsing as Sequence
  Labeling](https://www.aclweb.org/anthology/papers/N/N19/N19-1077/). Michalina
  Strzyz, David Vilares, Carlos Gómez-Rodríguez, 2019, Proceedings of
  the 2019 Conference of the North American Chapter of the Association
  for Computational Linguistics: Human Language Technologies

## Issues

You can report bugs and feature requests in the [sticker issue
tracker](https://github.com/stickeritis/sticker/issues).

## License

sticker is licensed under the [Blue Oak Model License version
1.0.0](LICENSE.md). The Tensorflow protocol buffer definitions in
`tf-proto` are licensed under the Apache License version 2.0. The
[list of contributors](CONTRIBUTORS) is also available.

## Credits

* sticker is developed by Daniël de Kok & Tobias Pütz.
* The Python precursor to sticker was developer by Erik Schill.
* Sebastian Pütz and Patricia Fischer reviewed a lot of code across
  the sticker projects.
