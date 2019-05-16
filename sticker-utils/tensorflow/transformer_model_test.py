import unittest

import numpy as np
import tensorflow as tf

from transformer_model import split_heads, self_attention


class TransformerTest(unittest.TestCase):
    def setUp(self):
        self.num_heads = 2
        self.inputs = tf.convert_to_tensor([
            # batch
            [  # position
                # features
                [1., 1, 1, 1],
                [1, 1, -1, -1],
                [0, 0, 0, 0]
            ],
            [
                [-2, 1, 2, 1],
                [2, 0, 1, -1],
                [1, 2, 1, 3]
            ],
        ], dtype=tf.float32)

        # manually calculated result of:
        # split_heads(self.inputs) @ split_heads(self.inputs).T
        self.calculated_scores = np.array([
            [  # head 1
                [
                    [2, 2, 0],
                    [2, 2, 0],
                    [0, 0, 0],
                ],
                # head 2
                [
                    [2, -2, 0],
                    [-2, 2, 0],
                    [0, 0, 0],
                ],
            ],
            [  # head 1
                [
                    [5, -4, 0],
                    [-4, 4, 2],
                    [0, 2, 5],
                ],
                # head 2
                [
                    [5, 1, 5],
                    [1, 2, -2],
                    [5, -2, 10],
                ],
            ]
        ], dtype=np.float32)

        scale = (self.num_heads ** -0.5)
        self.scaled_calculated_scores = self.calculated_scores * scale

    def run_in_session(self, args):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(args)
        return result

    def test_split_heads(self):
        targets = np.array([
            # batch
            [  # heads
                [  # positions
                    # features
                    [1, 1],
                    [1, 1],
                    [0, 0]

                ],
                [
                    [1, 1],
                    [-1, -1],
                    [0, 0]
                ]
            ],
            [
                [
                    [-2, 1],
                    [2, 0],
                    [1, 2]
                ],
                [
                    [2, 1],
                    [1, -1],
                    [1, 3]
                ]
            ]
        ], dtype=np.float32)

        results = self.run_in_session(split_heads(self.inputs, 2))
        np.testing.assert_equal(results, targets)

    def test_self_attention_inactive_zero(self):
        # test self attention with examples with lengths 2 and 3
        # the inactive steps are set to zero
        values = split_heads(self.inputs, self.num_heads)

        scaled_scores = self.scaled_calculated_scores

        # mask inactive steps
        scaled_scores[0, :, :, -1] = -1e38

        with tf.device("cpu:0"):
            # identity initializer to make results tractable
            with tf.variable_scope("", initializer=tf.initializers.identity):
                outputs, scores = self_attention(self.inputs, [2, 3], self.num_heads)

            outputs, scores, target_scores, values = self.run_in_session(
                [outputs, scores, tf.nn.softmax(scaled_scores), values])
            # Check masked scores are 0
            np.testing.assert_allclose(target_scores[0, :, :, -1], 0.)
            np.testing.assert_allclose(scores[0, :, :, -1], 0.)
            # Check scores
            #
            # set rtol to 1e-6 due to numerical instabilities.
            # calculation is correct with float64
            np.testing.assert_allclose(scores, target_scores, rtol=1e-6)

            # Check outputs
            targets = self.calculate_target_outputs(target_scores, values)
            # set rtol to 1e-6 due to numerical instabilities.
            # calculation is correct with float64
            np.testing.assert_allclose(outputs, targets, rtol=1e-6)

    def test_self_attention_inactive_value(self):
        # test self attention with examples with lengths 2 and 2
        # the inactive steps of example 1 are zero, of 2 non-zero
        values = split_heads(self.inputs, self.num_heads)

        # scale by 1 / sqrt(n_heads)
        scaled_scores = self.scaled_calculated_scores

        # mask inactive steps
        scaled_scores[:, :, :, -1] = -1e38

        with tf.device("cpu:0"):
            # identity initializer to make results tractable
            with tf.variable_scope("", initializer=tf.initializers.identity):
                outputs, scores = self_attention(self.inputs, [2, 2], self.num_heads)

            outputs, scores, target_scores, values = self.run_in_session(
                [outputs, scores, tf.nn.softmax(scaled_scores), values])
            # Check masked scores are 0
            np.testing.assert_allclose(target_scores[:, :, :, -1], 0.)
            np.testing.assert_allclose(scores[:, :, :, -1], 0.)

            # Check scores
            #
            # set rtol to 1e-6 due to numerical instabilities.
            # calculation is correct with float64
            np.testing.assert_allclose(scores, target_scores, rtol=1e-6)

            targets = self.calculate_target_outputs(target_scores, values)

            # set rtol to 1e-6 due to numerical instabilities.
            # calculation is correct with float64
            np.testing.assert_allclose(outputs, targets, rtol=1e-6)

    def calculate_target_outputs(self, target_scores, values):
        """
        Applies the attention scores `target_scores` to `values`.
        `target_scores` has shape [batch, heads, len, len]
        `values` has shape [batch, heads, len, head_dim]
        We calculate matrix multiplication between the inner matrices.
        `targets_scores[0,0]` @ `values[0,0]`
        `targets_scores[0,1]` @ `values[0,1]`
         .
         .
        Then we concatenate the per head results to restore shape:
        [batch, len, head_dim * 2].

        :param target_scores: float tensor of shape [batch, heads, len, len]
        :param values: float tensor of shape [batch, heads, len, head_dim]
        :return: a float tensor with applied attention of shape [batch, len, head_dim*2]
        """
        target_output = []
        # loop over batch
        for i in range(values.shape[0]):
            example_target = []
            # loop over heads
            for j in range(values.shape[1]):
                # apply attention scores of head j of example i to values
                # at head j of example i
                o = target_scores[i, j] @ values[i, j]
                example_target.append(o)
            # restore [batch, length, head_dim * 2] shape
            target_output.append(np.hstack(example_target))
        targets = np.array(target_output)
        return targets
