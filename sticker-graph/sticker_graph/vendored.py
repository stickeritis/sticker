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
