import math

import tensorflow as tf
from tensorflow.contrib.layers import layer_norm

from sticker_graph.model import Model


def split_heads(t, num_heads):
    """
    Splits the last dimension of `t` into `num_heads` dimensions and
    reorders dimensions to [batch, num_heads, length, head].

    :param t: tensor to be split
    :param num_heads: number of attention heads, needs to be a divisor of
            t.shape[-1]
    :return: reordered `t` with shape: [batch, num_heads, length, head]
    """
    shape = tf.shape(t)
    batch_size = shape[0]  # batch dimension
    length = shape[1]  # length dimension
    hidden_size = t.shape[-1]  # feature dimension needs to known statically
    error_msg = "n_heads needs to be a divisor of t.shape[-1]! heads: {}, " \
                "t.shape[-1]: {}".format(num_heads, hidden_size)
    assert hidden_size % num_heads == 0, error_msg

    head_size = hidden_size // num_heads  # dimensions per head

    # reshape to [batch, length, num_heads, head_size]
    b_l_h_f = tf.reshape(t, [batch_size, length, num_heads, head_size])

    # transpose to [batch, num_heads, length, head_size] so that attention is
    # across length dimension
    return tf.transpose(b_l_h_f, [0, 2, 1, 3])


def self_attention(inputs, seq_lens, num_heads):
    queries = tf.layers.dense(inputs, inputs.shape[-1], use_bias=False)
    keys = tf.layers.dense(inputs, inputs.shape[-1], use_bias=False)
    values = tf.layers.dense(inputs, inputs.shape[-1], use_bias=False)

    queries = split_heads(queries, num_heads=num_heads)
    keys = split_heads(keys, num_heads=num_heads)
    values = split_heads(values, num_heads=num_heads)

    # dot product + scale queries
    scale = tf.constant(num_heads ** -0.5)
    queries = queries * scale
    scores = tf.matmul(queries, keys, transpose_b=True)

    # we only care about not drawing information from inactive timesteps
    # inactive timesteps that draw information from active ones don't influence
    # results
    mask_s = tf.logical_not(tf.sequence_mask(seq_lens, tf.shape(inputs)[1]))
    mask_s = tf.to_float(mask_s[:, tf.newaxis, tf.newaxis]) * -1e38
    scores += mask_s

    scores = tf.nn.softmax(scores)

    # apply scores to values
    heads = tf.matmul(scores, values)

    # restore [batch, length, num_heads, head] order and rejoin num_heads and head
    heads = tf.reshape(tf.transpose(heads, [0, 2, 1, 3]), tf.shape(inputs))

    outputs = tf.layers.dense(heads, inputs.shape[-1], use_bias=False)

    return outputs, scores


def residual_feedforward_block(inputs,
                               inner_hsize,
                               outer_hsize,
                               activation,
                               keep_prob_inner,
                               keep_prob_outer,
                               is_training):
    """
    Feedforward block of the transformer. This block upsamples the input
    through a dense non-linear layer before downsampling it to the input
    dimensionality. The output is the sum of the output of the down-
    sampling layer and the inputs.

    :param inputs: input
    :param inner_hsize: dimensionality of the non-linear upsampling layer
    :param activation: activation function of the upsampling layer
    :param keep_prob_inner: keep probability of the upsampling layer
    :param keep_prob_outer: keep probability of the downsampling layer
    :param is_training: boolean indicator whether dropout will be applied
    :return: dense(activation(dense(inputs))) + inputs
    """
    inputs = layer_norm(inputs, begin_norm_axis=-1)
    up = tf.layers.dense(inputs, inner_hsize, activation)
    up = tf.contrib.layers.dropout(up,
                                   keep_prob=keep_prob_inner,
                                   is_training=is_training)
    down = tf.layers.dense(up, units=outer_hsize, use_bias=False)
    down = tf.contrib.layers.dropout(down,
                                     keep_prob=keep_prob_outer,
                                     is_training=is_training)
    return down + inputs


def self_attention_block(inputs,
                         seq_lens,
                         keep_prob_attention,
                         num_heads,
                         is_training):
    """
    Self attention block of the transformer. This block normalizes its inputs
    before applying multi-headed self-attention. The output a tuple. The first
    is the sum of the output of the self-attention layer and the inputs. The
    second are the alignment scores of the attention heads.
    
    :param inputs: input
    :param seq_lens: sequence lengths, used to mask inactive timesteps
    :param keep_prob_attention: keep probability of the attention output 
    :param num_heads: number of heads of the multi-headed attention
    :param is_training: boolean indicator whether dropout will be applied
    :return: self_attention(layer_norm(inputs)) + inputs, alignments
    """
    inputs = layer_norm(inputs, begin_norm_axis=-1)
    attention_output, scores = self_attention(inputs=inputs,
                                              seq_lens=seq_lens,
                                              num_heads=num_heads)
    attention_output = tf.contrib.layers.dropout(attention_output,
                                                 keep_prob=keep_prob_attention,
                                                 is_training=is_training)
    return attention_output + inputs, scores


def gelu(t):
    """
    Fast approximation of gaussian grror linear unit (GELU)
    https://github.com/hendrycks/GELUs

    :param t: tensor to apply gelu to
    :return: gelu(t)
    """
    return tf.nn.sigmoid(1.702 * t) * t


def build_encoder(inputs,
                  seq_lens,
                  activation,
                  config,
                  is_training):
    """
    Builds the transformer encoder. The encoder consists of multiple layers.
    Each layer consists of a self-attention block, followed by a residual
    feed-forward block. The output is a tuple. The first item a list with
    length n_layers containing the outputs of each layer. The second item
    is a list with length n_layers containing the alignment scores of the
    attention heads.

    :param inputs: Inputs
    :param seq_lens: Sequence length used to mask the self attention.
    :param activation: activation function of the inner feed-forward layer
    :param config: config object holding hyperparameters
    :param is_training: bool indicator whether dropout should be applied
    :return: tuple(list(encoder_states), list(attention_scores))
    """
    inner_hsize = config.inner_hsize
    outer_hsize = config.outer_hsize
    keep_prob_inner = config.keep_prob_inner
    keep_prob_outer = config.keep_prob_outer
    keep_prob_attention = config.keep_prob_attention
    num_layers = config.num_layers
    num_heads = config.num_heads

    states = inputs
    encoder = []
    alignments = []
    for n in range(num_layers):
        with tf.variable_scope("layer_{}".format(n)):
            with tf.variable_scope("self_attention"):
                states, scores = self_attention_block(states,
                                                      seq_lens=seq_lens,
                                                      keep_prob_attention=keep_prob_attention,
                                                      num_heads=num_heads,
                                                      is_training=is_training)

            with tf.variable_scope("feed_forward"):
                states = residual_feedforward_block(states,
                                                    inner_hsize=inner_hsize,
                                                    outer_hsize=outer_hsize,
                                                    activation=activation,
                                                    keep_prob_inner=keep_prob_inner,
                                                    keep_prob_outer=keep_prob_outer,
                                                    is_training=is_training)

        alignments.append(scores)
        encoder.append(states)

    # normalize output of the last layer
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L233
    encoder[-1] = layer_norm(encoder[-1], begin_norm_axis=-1)
    return encoder, alignments


def sinusoid(max_time_batch, depth):
    """
    Returns a mixture of sine and cosine as a timing signal.

    https://github.com/tensorflow/tensor2tensor/blob/9e0a894034d8090892c238df1bd9bd3180c2b9a3/tensor2tensor/layers/common_attention.py#L398
    :param max_time_batch: maximum number of timesteps in the batch
    :param depth: depth of the feature dimension
    :return: the sinusoidal time signal
    """
    single_wave_dims = depth // 2
    time_range = tf.cast(tf.range(max_time_batch) + 1, tf.float32)

    log_timescale_increment = math.log(float(1e4)) / (int(single_wave_dims) - 1)

    n_timescales = tf.cast(tf.range(single_wave_dims), tf.float32)
    inv_timescales = tf.exp(n_timescales * -log_timescale_increment)
    scaled_time = time_range[:, tf.newaxis] * inv_timescales[tf.newaxis,]

    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(depth, 2)]])[tf.newaxis,]

    return signal


def learned_positionals(config, max_time_batch, depth):
    """
    Embedding lookup to retrieve positional embeddings. Relies on
    `config.max_position` to determine the largest learnable position,
    everything beyond that is clipped and gets the same representation.

    This may be unproblematic for some use cases, in others it will
    influence performance.

    :param config: config object holding hyperparameters
    :param max_time_batch: maximum number of timesteps in the batch
    :param depth: depth of the feature dimension
    :return: the embedding lookup of the time signal
    """
    position_embs = tf.get_variable('positional_embeddings',
                                    dtype=tf.float32,
                                    shape=[config.max_position, depth])
    positions = tf.range(0, max_time_batch)
    clipped_positions = tf.clip_by_value(positions, 0, config.max_position)
    positions = tf.nn.embedding_lookup(position_embs, clipped_positions)
    return positions


class TransformerModel(Model):
    def __init__(self, config, shapes):
        super(TransformerModel, self).__init__(config, shapes)

        self.setup_placeholders()

        if config.activation == 'relu':
            activation = tf.nn.relu
        elif config.activation == 'gelu':
            activation = gelu
        else:
            raise NotImplementedError('Activation %s is not available.'
                                      % config.activation)

        inputs = tf.contrib.layers.dropout(
            self.inputs,
            keep_prob=config.keep_prob_input,
            is_training=self.is_training)

        if not config.pass_inputs:
            inputs = tf.layers.dense(inputs, config.outer_hsize, activation)
        else:
            error_msg = "With '--pass_inputs' the last input dimension has " \
                        "to match '--outer_hsize'. OUTER_HSIZE: %d, " \
                        "input[-1]: %d" % (config.outer_hsize, inputs.shape[-1])

            assert inputs.shape[-1] == config.outer_hsize, error_msg

        # add time signal
        max_time_batch, depth = tf.shape(self.inputs)[1], inputs.shape[-1]
        if config.embed_time:
            inputs += learned_positionals(config, max_time_batch, depth)
        else:
            inputs += sinusoid(max_time_batch, depth)

        hidden_states, alignments = build_encoder(inputs=inputs,
                                                  seq_lens=self.seq_lens,
                                                  activation=activation,
                                                  config=config,
                                                  is_training=self.is_training, )

        logits = self.affine_transform(
            "tag", hidden_states[-1], shapes['n_labels'])
        if config.crf:
            loss, transitions = self.crf_loss(
                "tag", logits, self.tags)
            predictions, top_k_predictions = self.crf_predictions(
                "tag", logits, transitions)
        else:
            loss = self.masked_softmax_loss(
                "tag", logits, self.tags, self.mask)
            predictions = self.predictions("tag", logits)
            self.top_k_predictions("tag", logits, config.top_k)

        acc = self.accuracy("tag", predictions, self.tags)

        # Optimization with gradient clipping. Consider making the gradient
        # norm a placeholder as well.
        lr = tf.placeholder(tf.float32, [], "lr")

        optimizer = tf.train.AdamOptimizer(lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, gradient_norm = tf.clip_by_global_norm(gradients, 2.5)

        train_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(
            zip(gradients, variables), name="train", global_step=train_step)

        self.create_summary_ops(acc, gradient_norm, loss, lr)