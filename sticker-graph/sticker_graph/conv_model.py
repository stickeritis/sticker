from enum import Enum

import tensorflow as tf
from sticker_graph.model import Model
from sticker_graph.weight_norm import WeightNorm


class Sharing(Enum):
    none = 1
    initial = 2
    succeeding = 3


def mask_layer(layer, mask):
    return tf.multiply(
        tf.broadcast_to(
            tf.expand_dims(
                mask, -1), tf.shape(layer)), layer)


def dilated_convolution(
        x,
        n_outputs,
        kernel_size,
        n_levels,
        is_training,
        mask,
        glu=True,
        keep_prob=1.0):
    layer = x

    for i in range(n_levels):
        # Only use sharing for layers 1 and up. Layer 0 cannot use sared parameters:
        #
        # - It transforms word embeddings into the hidden representation,
        #   whereas subsequent layers transform hidden representations to
        #   hidden representations.
        # - The input size may differ from the output size.
        if i == 0:
            sharing = Sharing.none
        elif i == 1:
            sharing = Sharing.initial
        else:
            sharing = Sharing.succeeding

        dilation = 2 ** i
        layer = residual_block(
            layer,
            n_outputs,
            kernel_size,
            dilation,
            is_training=is_training,
            mask=mask,
            glu=glu,
            keep_prob=keep_prob,
            sharing=sharing)

    # Mask after last convolution. This is only necessary for models that
    # apply transformations across time steps after the diluted convolutions.
    # But masking is cheap, so better safe than sorry.
    layer = mask_layer(layer, mask)

    return layer


def residual_block(
        x,
        n_outputs,
        kernel_size,
        dilation,
        is_training,
        mask,
        glu=True,
        keep_prob=1.0,
        sharing=Sharing.none):
    if sharing == Sharing.initial or sharing == Sharing.succeeding:
        suffix = "shared"
    else:
        suffix = "unshared"

    with tf.variable_scope("conv1-%s" % suffix, reuse=sharing == Sharing.succeeding):
        conv1 = residual_unit(
            x,
            n_outputs,
            kernel_size,
            dilation,
            is_training,
            mask=mask,
            glu=glu,
            keep_prob=keep_prob)
    with tf.variable_scope("conv2-%s" % suffix, reuse=sharing == Sharing.succeeding):
        conv2 = residual_unit(
            conv1,
            n_outputs,
            kernel_size,
            dilation,
            is_training,
            mask=mask,
            glu=glu,
            keep_prob=keep_prob)

    if x.get_shape()[2] != n_outputs:
        # Note: biases could change padding timesteps, but the next layer will mask
        #       the resulting sequence.
        x = tf.layers.Conv1D(n_outputs, 1)(x)

    return x + conv2


def residual_unit(
        x,
        n_outputs,
        kernel_size,
        dilation,
        is_training,
        mask,
        glu=True,
        keep_prob=1.0):
    if glu:
        # For GLU we need the hidden representation, plus an equal number
        # of parameters for weighting the hidden representation.
        n_outputs *= 2

    # Mask inactive time steps. This is necessary, because convolutions make
    # the padding non-zero (through past timesteps). In later convolutions,
    # these updated paddings would then influence time steps before the
    # padding.
    x = mask_layer(x, mask)
    conv = WeightNorm(
        tf.layers.Conv1D(
            n_outputs,
            kernel_size,
            dilation_rate=dilation,
            padding="same"))(x)

    if glu:
        left, right = tf.split(conv, num_or_size_splits=2, axis=2)
        left = tf.sigmoid(left)
        conv = tf.multiply(left, right)
    else:
        conv = tf.nn.relu(conv)

    # Spatial dropout
    conv = tf.contrib.layers.dropout(
        conv,
        keep_prob=keep_prob,
        noise_shape=[
            tf.shape(conv)[0],
            tf.constant(1),
            tf.shape(conv)[2]],
        is_training=is_training)

    return conv


class ConvModel(Model):
    def __init__(
            self,
            args,
            shapes):
        super(ConvModel, self).__init__(args, shapes)

        self.setup_placeholders()

        inputs = tf.contrib.layers.dropout(
            self.inputs,
            keep_prob=args.keep_prob_input,
            is_training=self.is_training)

        hidden_states = dilated_convolution(
            inputs,
            args.hidden_size,
            kernel_size=args.kernel_size,
            n_levels=args.levels,
            is_training=self.is_training,
            glu=not args.relu,
            keep_prob=args.keep_prob,
            mask=self.mask)

        # Normalize hidden layers, seems to speed up convergence.
        hidden_states = tf.contrib.layers.layer_norm(
            hidden_states, begin_norm_axis=-1)

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

        # Optimization with gradient clipping. Consider making the gradient
        # norm a placeholder as well.
        lr = tf.placeholder(tf.float32, [], "lr")
        optimizer = tf.train.AdamOptimizer(lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, gradient_norm = tf.clip_by_global_norm(gradients, 1.0)

        train_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(
            zip(gradients, variables), name="train", global_step=train_step)

        self.create_summary_ops(acc, gradient_norm, loss, lr)
