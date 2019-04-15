import tensorflow as tf


class Model:
    def __init__(self, config, shapes):
        self._config = config
        self._shapes = shapes

    def accuracy(self, prefix, predictions, labels):
        correct = tf.equal(predictions, labels)

        # Mask inactive timesteps
        correct = tf.multiply(self.mask, tf.cast(correct, tf.float32))

        # Compensate for inactive time steps.
        correct = tf.reshape(correct, [-1])
        correct = tf.truediv(correct, tf.reduce_mean(self.mask))

        return tf.reduce_mean(correct, name="%s_accuracy" % prefix)

    def affine_transform(self, prefix, x, n_outputs):
        input_size = x.get_shape()[-1]
        batch_size = tf.shape(x)[0]

        w = tf.get_variable(
            "%s_linear_w" % prefix, [
                input_size, n_outputs])
        b = tf.get_variable("%s_linear_b" % prefix, [n_outputs])

        x = tf.reshape(x, [-1, input_size])

        return tf.reshape(
            tf.nn.xw_plus_b(
                x, w, b), [
                batch_size, -1, n_outputs])

    def masked_softmax_loss(self, prefix, logits, labels, mask):
        # Compute losses
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

        # Mask inactive time steps.
        losses = tf.multiply(mask, losses)

        # Compensate for inactive time steps.
        losses = tf.truediv(losses, tf.reduce_mean(mask))

        return tf.reduce_mean(losses, name="%s_loss" % prefix)

    def crf_loss(self, prefix, logits, labels):
        with tf.variable_scope("%s_crf" % prefix):
            (loss, transitions) = tf.contrib.crf.crf_log_likelihood(
                logits, labels, self.seq_lens)
        return tf.reduce_mean(-loss, name="%s_loss" % prefix), transitions

    def crf_predictions(self, prefix, logits, transitions):
        predictions, _ = tf.contrib.crf.crf_decode(
            logits, transitions, self.seq_lens)
        return tf.identity(predictions, name="%s_predictions" % prefix)

    def predictions(self, prefix, logits):
        # Get the best label, excluding padding.
        best_label = tf.argmax(
            logits[:, :, 1:],
            axis=2,
            output_type=tf.dtypes.int32)

        # Exclusion of padding shifts all classes by one.
        return tf.add(best_label, 1, name="%s_predictions" % prefix)

    def top_k_predictions(self, prefix, logits, k):
        # Get the best label, excluding padding.
        (_, best_labels) = tf.math.top_k(
            logits[:, :, 1:],
            k=k)

        # Exclusion of padding shifts all classes by one.
        return tf.add(best_labels, 1, name="%s_top_k_predictions" % prefix)

    def setup_placeholders(self):
        self._is_training = tf.placeholder(tf.bool, [], "is_training")

        self._tags = tf.placeholder(
            tf.int32, name="tags", shape=[
                None, None])

        self._inputs = tf.placeholder(
            tf.float32,
            shape=[
                None,
                None,
                self.shapes['token_embed_dims'] +
                self.shapes['tag_embed_dims']],
            name="inputs")

        self._seq_lens = tf.placeholder(
            tf.int32, [None], name="seq_lens")

        # Compute mask
        self._mask = tf.sequence_mask(
            self.seq_lens, maxlen=tf.shape(
                self.inputs)[1], dtype=tf.float32)

    @property
    def config(self):
        return self._config

    @property
    def is_training(self):
        return self._is_training

    @property
    def mask(self):
        return self._mask

    @property
    def seq_lens(self):
        return self._seq_lens

    @property
    def tags(self):
        return self._tags

    @property
    def inputs(self):
        return self._inputs

    @property
    def shapes(self):
        return self._shapes
