# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools

import tensorflow as tf


def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None,
                                    residual_connections=False):
    """
    NOTE:
    This is a modified copy of tf.contrib.rnn.stack_bidirectional_dynamic_rnn
    that adds the option to have residual skip connections. It has been taken
    from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/rnn/python/ops/rnn.py.

    If residual connections is True, the input of a layer is summed with the
    output of the layer. In order to allow for inputs with other dimensionality
    than that of the concatenated RNN states, the input to the first layer is
    not summed to its output.
    """
    if not cells_fw:
        raise ValueError(
            "Must specify at least one fw cell for BidirectionalRNN.")
    if not cells_bw:
        raise ValueError(
            "Must specify at least one bw cell for BidirectionalRNN.")
    if not isinstance(cells_fw, list):
        raise ValueError(
            "cells_fw must be a list of RNNCells (one per layer).")
    if not isinstance(cells_bw, list):
        raise ValueError(
            "cells_bw must be a list of RNNCells (one per layer).")
    if len(cells_fw) != len(cells_bw):
        raise ValueError(
            "Forward and Backward cells must have the same depth.")
    if (initial_states_fw is not None and
            (not isinstance(initial_states_fw, list) or
             len(initial_states_fw) != len(cells_fw))):
        raise ValueError(
            "initial_states_fw must be a list of state tensors (one per layer).")
    if (initial_states_bw is not None and
            (not isinstance(initial_states_bw, list) or
             len(initial_states_bw) != len(cells_bw))):
        raise ValueError(
            "initial_states_bw must be a list of state tensors (one per layer).")

    states_fw = []
    states_bw = []
    prev_layer = inputs

    with tf.compat.v1.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with tf.compat.v1.variable_scope("cell_%d" % i):
                shortcut = prev_layer
                outputs, (state_fw, state_bw) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype,
                    time_major=time_major)
                # Concat the outputs to create the new input.
                prev_layer = tf.concat(outputs, 2)
                if i != 0 and residual_connections:
                    prev_layer += shortcut

            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


def _create_file_writer_generic_type(logdir,
                                     name="logdir",
                                     max_queue=None,
                                     flush_millis=None,
                                     filename_suffix=None):
    """
    This method mirrors `tensorflow.contrib.summary.create_file_writer`. Unlike
    `summary.create_file_writer`, this method accepts a placeholder as `logdir`.
    """
    from tensorflow.python.ops.gen_summary_ops import create_summary_file_writer

    logdir = tf.convert_to_tensor(value=logdir)
    with tf.device("cpu:0"):
        if max_queue is None:
            max_queue = tf.constant(10)
        if flush_millis is None:
            flush_millis = tf.constant(2 * 60 * 1000)
        if filename_suffix is None:
            filename_suffix = tf.constant(".v2")

    from pkg_resources import parse_version
    if parse_version(tf.version.VERSION) >= parse_version("1.14"):
        from tensorflow.python.ops.summary_ops_v2 import ResourceSummaryWriter
        return ResourceSummaryWriter(
            shared_name=name,
            init_op_fn=functools.partial(create_summary_file_writer,
                                         logdir=logdir,
                                         max_queue=max_queue,
                                         flush_millis=flush_millis,
                                         filename_suffix=filename_suffix))
    else:
        from tensorflow.python.ops.summary_ops_v2 import _make_summary_writer
        return _make_summary_writer(
            name,
            create_summary_file_writer,
            logdir=logdir,
            max_queue=max_queue,
            flush_millis=flush_millis,
            filename_suffix=filename_suffix)
