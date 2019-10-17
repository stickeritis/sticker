# Training a sticker model

In order to train a model, a model configuration file is needed. This file
describes settings such as which embeddings should be used. Sample
configuration files for various tasks can be found in the [examples](examples/)
directory. Given a configuration file, the first step is to create a shapes
file.

~~~shell
$ sticker prepare postag.conf train.conll postag.shapes
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
$ sticker train postag.conf train.conll validation.conll
~~~

The models are quite sensitive to the learning rate and may diverge
when the learning rate is to high. The default learning rate is *0.01*
and can be set using the `lr` option. For example, to use a learning
rate of *0.001*, use:

~~~shell
$ sticker train --lr 0.001 postag.conf train.conll validation.conll
~~~

The training procedure will output the best epoch. Update the
`parameters` setting in the configuration file to use that epoch.

______

#### Tensorboard

The training supports
[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
logging. In order to track the training
in a browser, start the training with the `--logdir` option set to a directory 
with write access: 

~~~shell
$ sticker train postag.conf train.conll validation.conll --logdir pos_logdir
~~~

This will write the Tensorboard summaries to the directory `pos_logdir` in
the current folder.

In order to visualize the summaries, open another shell and point 
Tensorboard to the logdir:

~~~shell
$ tensorboard --logdir pos_logdir
~~~

After a brief moment, the address (e.g. localhost:6006) under which the 
summaries can be viewed in a browser should appear.
