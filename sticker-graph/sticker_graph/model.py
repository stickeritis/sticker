import tensorflow as tf

from sticker_graph.rnn import bidi_rnn_layers


class Model:
    def __init__(self, args, shapes):
        self._args = args
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

        w = tf.compat.v1.get_variable(
            "%s_linear_w" % prefix, [
                input_size, n_outputs])
        b = tf.compat.v1.get_variable("%s_linear_b" % prefix, [n_outputs])

        x = tf.reshape(x, [-1, input_size])

        return tf.reshape(
            tf.compat.v1.nn.xw_plus_b(
                x, w, b), [
                batch_size, -1, n_outputs])

    def decode_word(self, word, max_subword_len):
        bytes = tf.io.decode_raw(word, tf.uint8)[:max_subword_len]
        return tf.pad(tensor=bytes, paddings=[[0, max_subword_len - tf.shape(input=bytes)[0]]])

    def decode_words(self, words, max_subword_len):
        shape = tf.shape(words)

        # Flatten, because map_fn only works on axis 0
        flat = tf.reshape(words, [-1])

        # Decode words.
        decoded = tf.map_fn(
            lambda x: self.decode_word(
                x,
                max_subword_len),
            flat,
            dtype=tf.uint8)

        # Restore shape
        return tf.reshape(decoded, [shape[0], shape[1], max_subword_len])

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
        with tf.compat.v1.variable_scope("%s_crf" % prefix):
            (loss, transitions) = tf.contrib.crf.crf_log_likelihood(
                logits, labels, self.seq_lens)
        return tf.reduce_mean(-loss, name="%s_loss" % prefix), transitions

    def crf_predictions(self, prefix, logits, transitions):
        predictions, _ = tf.contrib.crf.crf_decode(
            logits, transitions, self.seq_lens)
        top_k_predictions = tf.expand_dims(predictions, 2)

        # We don't get per-item probabilities, so just return ones.
        top_k_probs = tf.ones(tf.shape(top_k_predictions), tf.float32,
                              name="%s_top_k_probs" % prefix)

        return predictions, tf.identity(
            top_k_predictions, name="%s_top_k_predictions" %
            prefix)

    def predictions(self, prefix, logits):
        # Get the best label, excluding padding.
        best_label = tf.argmax(
            logits[:, :, 1:],
            axis=2,
            output_type=tf.dtypes.int32)

        # Exclusion of padding shifts all classes by one.
        return tf.add(best_label, 1, name="%s_predictions" % prefix)

    def top_k_predictions(self, prefix, logits, k):
        probs = tf.nn.softmax(logits)

        # Exclude padding.
        non_pad_probs = probs[:, :, 1:]

        # Get the best label, excluding padding.
        best_probs, best_labels = tf.nn.top_k(
            non_pad_probs,
            k=k)

        probs = tf.identity(best_probs, name="%s_top_k_probs" % prefix)
        labels = tf.add(best_labels, 1, name="%s_top_k_predictions" % prefix)

        return probs, labels

    def setup_inputs(self):
        # Word/tag embeddings
        inputs = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                None,
                None,
                self.shapes['token_embed_dims'] +
                self.shapes['tag_embed_dims']],
            name="inputs")
        inputs = tf.contrib.layers.dropout(
            inputs,
            keep_prob=self.args.keep_prob_input,
            is_training=self.is_training)

        if self.shapes['subwords']:
            # Forms for subwords
            self._subwords = tf.compat.v1.placeholder(
                tf.string,
                shape=[
                    None,
                    None],
                name="subwords")
            inputs = tf.concat([inputs, self.subword_reprs()], axis=-1)

        self._inputs = inputs

    def setup_placeholders(self):
        self._is_training = tf.compat.v1.placeholder(
            tf.bool, [], "is_training")

        self._tags = tf.compat.v1.placeholder(
            tf.int32, name="tags", shape=[
                None, None])

        self.setup_inputs()

        self._seq_lens = tf.compat.v1.placeholder(
            tf.int32, [None], name="seq_lens")

        # Compute mask
        self._mask = tf.sequence_mask(
            self.seq_lens, maxlen=tf.shape(
                self.inputs)[1], dtype=tf.float32)

    def subword_reprs(self):
        # Convert strings to a byte tensor.
        #
        # Shape: [batch_size, seq_len, subword_len]
        with tf.device("/cpu:0"):
            subword_bytes = tf.cast(
                self.decode_words(
                    self.subwords,
                    self.args.subword_len),
                tf.int32)

        # Get the lengths of the subwords.
        # Shape: [batch_size, seq_len]
        with tf.device("/cpu:0"):
            subword_lens = tf.math.count_nonzero(
                subword_bytes, axis=-1, dtype=tf.int32)

        # Lookup byte embeddings, this results in a tensor of shape.
        #
        # Shape: [batch_size, seq_len, max_bytes_len, byte_embed_size]
        byte_embeds = tf.compat.v1.get_variable(
            "byte_embeds", [
                256, self.args.byte_embed_size])
        byte_reprs = tf.nn.embedding_lookup(byte_embeds, subword_bytes)

        byte_reprs = tf.contrib.layers.dropout(
            byte_reprs,
            keep_prob=self.args.keep_prob_input,
            is_training=self.is_training)

        # Prepare shape for applying the RNN:
        #
        # Shape: [batch_size * seq_len, max_bytes_len, byte_embed_size]
        bytes_shape = tf.shape(subword_bytes)
        byte_reprs = tf.reshape(
            byte_reprs, [-1, bytes_shape[2], self.args.byte_embed_size])
        byte_lens = tf.reshape(subword_lens, [-1])

        with tf.compat.v1.variable_scope("byte_rnn"):
            subword_reprs = bidi_rnn_layers(
                self.is_training,
                byte_reprs,
                num_layers=self.args.subword_layers,
                output_size=self.args.subword_hidden_size,
                output_keep_prob=self.args.subword_keep_prob,
                seq_lens=byte_lens,
                gru=self.args.subword_gru,
                residual_connections=self.args.subword_residual,
                return_sequences=False)

        return tf.reshape(subword_reprs, [bytes_shape[0], bytes_shape[1],
                                          subword_reprs.shape[-1]])

    def create_summary_ops(self, acc, grad_norm, loss, lr):
        step = tf.compat.v1.train.get_or_create_global_step()

        train_summaries = [tf.contrib.summary.scalar(name="loss",
                                                     tensor=loss,
                                                     step=step,
                                                     family="train"),
                           tf.contrib.summary.scalar(name="accuracy",
                                                     tensor=acc,
                                                     step=step,
                                                     family="train"),
                           tf.contrib.summary.scalar(name="learning rate",
                                                     tensor=lr,
                                                     step=step,
                                                     family="train")]

        if grad_norm is not None:
            train_summaries.append(
                tf.contrib.summary.scalar(
                    name="gradient_norm",
                    tensor=grad_norm,
                    step=step,
                    family="train"))

        val_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            name="val_global_step")
        with tf.control_dependencies([val_step.assign_add(1)]):
            val_step = tf.convert_to_tensor(val_step)
            val_summaries = [
                tf.contrib.summary.scalar(name="loss",
                                          tensor=loss,
                                          step=val_step,
                                          family="val"),
                tf.contrib.summary.scalar(name="acc",
                                          tensor=acc,
                                          step=val_step,
                                          family="val")]
        with tf.compat.v1.variable_scope("summaries"):
            self.train_summaries = tf.group(train_summaries, name="train")
            self.val_summaries = tf.group(val_summaries, name="val")

    @property
    def args(self):
        return self._args

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
    def subwords(self):
        return self._subwords

    @property
    def tags(self):
        return self._tags

    @property
    def inputs(self):
        return self._inputs

    @property
    def shapes(self):
        return self._shapes
