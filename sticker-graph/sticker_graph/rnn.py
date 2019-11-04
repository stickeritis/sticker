import tensorflow as tf

from sticker_graph.keras_vendored import GRU, LSTM


def bidi_rnn_layers(
        is_training,
        inputs,
        num_layers=1,
        output_size=50,
        output_keep_prob=1.0,
        gru=False,
        residual_connections=False,
        return_sequences=True,
        seq_lens=None):
    if gru:
        rnn_layer = GRU
    else:
        rnn_layer = LSTM

    # Compute mask
    mask = None
    if seq_lens is not None:
        mask = tf.sequence_mask(
            seq_lens, maxlen=tf.shape(
                inputs)[1])

    layer = inputs
    for i in range(num_layers):
        # Keep a reference to the previous layer for residual connections.
        prev_layer = layer

        layer_return_sequences = True
        if i == num_layers - 1:
            layer_return_sequences = return_sequences

        # Bidirectional RNN + state output dropout.
        layer = tf.compat.v2.keras.layers.Bidirectional(
            rnn_layer(
                output_size,
                return_sequences=layer_return_sequences))(
            layer,
            mask=mask)
        layer = tf.compat.v2.keras.layers.Dropout(
            1.0 -
            output_keep_prob)(
            layer,
            training=is_training)

        # Add a residual connection if requested. A residual connection
        # is not added for the first layer, since input/output sizes
        # may mismatch.
        if i != 0 and residual_connections:
            layer = layer + prev_layer

    return layer
