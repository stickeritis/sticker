import tensorflow as tf

import sticker_graph.vendored


def dropout_wrapper(
        cell,
        is_training,
        output_keep_prob=1.0,
        state_keep_prob=1.0):
    output_keep_prob = tf.cond(
        pred=is_training,
        true_fn=lambda: tf.constant(output_keep_prob),
        false_fn=lambda: tf.constant(1.0))
    state_keep_prob = tf.cond(
        pred=is_training,
        true_fn=lambda: tf.constant(state_keep_prob),
        false_fn=lambda: tf.constant(1.0))
    return tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        cell,
        output_keep_prob=output_keep_prob,
        state_keep_prob=state_keep_prob)


def bidi_rnn_layers(
        is_training,
        inputs,
        num_layers=1,
        output_size=50,
        output_keep_prob=1.0,
        state_keep_prob=1.0,
        seq_lens=None,
        gru=False,
        residual_connections=False):
    if gru:
        cell = tf.compat.v1.nn.rnn_cell.GRUCell
    else:
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell

    fw_cells = [
        dropout_wrapper(
            cell=cell(output_size),
            is_training=is_training,
            state_keep_prob=state_keep_prob,
            output_keep_prob=output_keep_prob) for i in range(num_layers)]

    bw_cells = [
        dropout_wrapper(
            cell=cell(output_size),
            is_training=is_training,
            state_keep_prob=state_keep_prob,
            output_keep_prob=output_keep_prob) for i in range(num_layers)]
    return sticker_graph.vendored.stack_bidirectional_dynamic_rnn(
        fw_cells,
        bw_cells,
        inputs,
        dtype=tf.float32,
        sequence_length=seq_lens,
        residual_connections=residual_connections)
