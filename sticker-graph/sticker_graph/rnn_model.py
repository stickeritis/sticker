import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from sticker_graph.model import Model
from sticker_graph.rnn import bidi_rnn_layers


class RNNModel(Model):
    def __init__(
            self,
            args,
            shapes):
        super(RNNModel, self).__init__(args, shapes)

        self.setup_placeholders()

        hidden_states = bidi_rnn_layers(
            self.is_training,
            self.inputs,
            num_layers=args.rnn_layers,
            output_size=args.hidden_size,
            output_keep_prob=args.keep_prob,
            seq_lens=self._seq_lens,
            gru=args.gru,
            residual_connections=args.residual)

        hidden_states = batch_norm(
            hidden_states,
            decay=0.98,
            scale=True,
            is_training=self.is_training,
            fused=False,
            updates_collections=None)

        logits = self.affine_transform(
            "tag", hidden_states, shapes['n_labels'])
        if args.crf:
            loss, transitions = self.crf_loss(
                "tag", logits, self.tags)
            predictions, top_k_predictions = self.crf_predictions(
                "tag", logits, transitions)
        else:
            loss = self.masked_softmax_loss(
                "tag", logits, self.tags, self.mask)
            predictions = self.predictions("tag", logits)
            self.top_k_predictions("tag", logits, args.top_k)

        acc = self.accuracy("tag", predictions, self.tags)

        lr = tf.compat.v1.placeholder(tf.float32, [], "lr")

        train_step = tf.compat.v1.train.get_or_create_global_step()
        optimizer = tf.compat.v1.train.AdamOptimizer(lr)
        if args.auto_mixed_precision:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        self._train_op = optimizer.minimize(loss, name="train", global_step=train_step)

        self.create_summary_ops(acc, None, loss, lr)
