# Using sticker

## Tagging local data

Given an existing model configuration such as `postag.conf`, you can
use `sticker-tag` to annotate data in CoNLL-X format:

~~~shell
$ sticker-tag postag.conf input.conll output.conll
~~~

When the input and output are not specified, `sticker-tag` will read
from the *standard input* and write to the *standard output*.

## Client/server

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
