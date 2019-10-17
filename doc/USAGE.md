# Using sticker

## Tagging local data

Given an existing model configuration such as `postag.conf`, you can
use `sticker tag` to annotate data in CoNLL-X format:

~~~shell
$ sticker tag postag.conf \
  --input input.conll --output output.conll
~~~

When the input and output are not specified, `sticker tag` will read
from the *standard input* and write to the *standard output*. If
you have multiple models, you can specify their configuration files
to form a pipeline. The output of the first model is passed to the
second model, etc. For example:

~~~shell
$ sticker tag postag.conf depparse.conf \
  --input input.conll --output output.conll
~~~

## Client/server

sticker can also run as a simple server:

~~~shell
$ sticker server postag.conf
~~~

This will load the model defined in `postag.conf` and then listen on a
socket on `localhost` port `4000`. You can then send data in CoNLL-X
format to this port and the sticker will return the annotated
data. The last chunk of data will only be written if the client shuts
down the writing end off their socker (see the `shutdown(2)` manual
page).
