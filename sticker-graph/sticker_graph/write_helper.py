import argparse
import tensorflow as tf
import toml


def read_shapes(args):
    with open(args.shape_file) as shapesfile:
        shapes = toml.loads(shapesfile.read())
    return shapes


def _create_file_writer_generic_type(logdir,
                                    name="logdir",
                                    max_queue=None,
                                    flush_millis=None,
                                    filename_suffix=None):
    """
    This method mirrors `tensorflow.contrib.summary.create_file_writer`. Unlike
    `summary.create_file_writer`, this method accepts a placeholder as `logdir`.
    """
    from tensorflow.python.ops.summary_ops_v2 import _make_summary_writer
    from tensorflow.python.ops.gen_summary_ops import create_summary_file_writer

    logdir = tf.convert_to_tensor(logdir)
    with tf.device("cpu:0"):
        if max_queue is None:
            max_queue = tf.constant(10)
        if flush_millis is None:
            flush_millis = tf.constant(2 * 60 * 1000)
        if filename_suffix is None:
            filename_suffix = tf.constant(".v2")
        return _make_summary_writer(
            name,
            create_summary_file_writer,
            logdir=logdir,
            max_queue=max_queue,
            flush_millis=flush_millis,
            filename_suffix=filename_suffix)


def create_graph(model, args):
    shapes = read_shapes(args)
    graph_filename = args.output_graph_file

    gpuopts = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    tfconfig = tf.ConfigProto(gpu_options=gpuopts)

    with tf.Graph().as_default(), tf.Session(config=tfconfig) as session:
        logdir = tf.placeholder(shape=[], name="logdir", dtype=tf.string)
        summary_writer = _create_file_writer_generic_type(logdir)

        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            with tf.variable_scope("model", reuse=None):
                model(args=args, shapes=shapes)

            tf.group(tf.contrib.summary.summary_writer_initializer_op(),
                     name="summary_init")
            tf.variables_initializer(tf.global_variables(), name='init')

            tf.train.Saver(tf.global_variables())

            serialized_graph = session.graph_def.SerializeToString()
            serialized_graph_tensor = tf.convert_to_tensor(serialized_graph)
            tf.contrib.summary.graph(serialized_graph_tensor, 0, name='graph_write')
            tf.train.write_graph(
                session.graph_def,
                './',
                graph_filename,
                as_text=False)


def get_common_parser():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument(
        'shape_file',
        metavar='SHAPE_FILE',
        type=str,
        help='shape file')
    parser.add_argument(
        'output_graph_file',
        metavar='OUTPUT_GRAPH_FILE',
        type=str,
        help='output graph file')
    parser.add_argument(
        "--crf",
        help="use CRF layer for classification",
        action="store_true")
    parser.add_argument(
        "--top_k",
        type=int,
        help="number of predictions to return",
        default=3)
    return parser
