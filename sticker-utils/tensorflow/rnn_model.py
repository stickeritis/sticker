#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from enum import Enum
from model import Model


def dropout_wrapper(cell, is_training, keep_prob):
    keep_prob = tf.cond(
        is_training,
        lambda: tf.constant(keep_prob),
        lambda: tf.constant(1.0))
    return tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=keep_prob)


def rnn_layers(
        is_training,
        inputs,
        num_layers=1,
        output_size=50,
        output_dropout=1,
        state_dropout=1,
        seq_lens=None,
        bidirectional=False,
        gru=False):
    if gru:
        cell = tf.nn.rnn_cell.GRUCell
    else:
        cell = tf.contrib.rnn.BasicLSTMCell

    forward_cell = cell(output_size)
    forward_cell = dropout_wrapper(
        cell=forward_cell,
        is_training=is_training,
        keep_prob=output_dropout)

    if not bidirectional:
        return tf.nn.dynamic_rnn(
            forward_cell,
            inputs,
            dtype=tf.float32,
            sequence_length=seq_lens)

    backward_cell = cell(output_size)
    backward_cell = dropout_wrapper(
        cell=backward_cell,
        is_training=is_training,
        keep_prob=output_dropout)

    return tf.nn.bidirectional_dynamic_rnn(
        forward_cell,
        backward_cell,
        inputs,
        dtype=tf.float32,
        sequence_length=seq_lens)


class RNNModel(Model):
    def __init__(
            self,
            config,
            shapes):
        super(RNNModel, self).__init__(config, shapes)

        self.setup_placeholders()

        inputs = tf.contrib.layers.dropout(
            self.inputs,
            keep_prob=config.keep_prob_input,
            is_training=self.is_training)

        (fstates, bstates), _ = rnn_layers(
            self.is_training,
            inputs,
            num_layers=config.rnn_layers,
            output_size=config.hidden_size,
            output_dropout=config.keep_prob,
            state_dropout=config.keep_prob,
            seq_lens=self._seq_lens,
            bidirectional=True,
            gru=config.gru)
        hidden_states = tf.concat([fstates, bstates], axis=2)

        hidden_states, _ = rnn_layers(self.is_training, hidden_states, num_layers=1, output_size=config.hidden_size,
                                      output_dropout=config.keep_prob,
                                      state_dropout=config.keep_prob, seq_lens=self._seq_lens, gru=config.gru)

        hidden_states = batch_norm(
            hidden_states,
            decay=0.98,
            scale=True,
            is_training=self.is_training,
            fused=False,
            updates_collections=None)

        logits = self.affine_transform(
            "tag", hidden_states, shapes['n_labels'])
        if config.crf:
            loss, transitions = self.crf_loss(
                "tag", logits, self.tags)
            predictions = self.crf_predictions(
                "tag", logits, transitions)
        else:
            loss = self.masked_softmax_loss(
                "tag", logits, self.tags, self.mask)
            predictions = self.predictions("tag", logits)
            self.top_k_predictions("tag", logits, config.top_k)


        self.accuracy("tag", predictions, self.tags)

        lr = tf.placeholder(tf.float32, [], "lr")
        self._train_op = tf.train.AdamOptimizer(
            lr).minimize(loss, name="train")
