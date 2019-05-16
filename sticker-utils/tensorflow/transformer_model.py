import tensorflow as tf

from model import Model


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
