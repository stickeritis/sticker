import argparse
import sys
import tensorflow as tf
import toml

import sticker_graph.vendored


def read_shapes(args):
    with open(args.shape_file) as shapesfile:
        shapes = toml.loads(shapesfile.read())
    return shapes


def create_graph(model, args):

    shapes = read_shapes(args)
    graph_filename = args.output_graph_file

    gpuopts = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    tfconfig = tf.ConfigProto(gpu_options=gpuopts)

    with tf.Graph().as_default(), tf.Session(config=tfconfig) as session:
        write_args(model, args)

        logdir = tf.placeholder(shape=[], name="logdir", dtype=tf.string)
        summary_writer = sticker_graph.vendored._create_file_writer_generic_type(logdir)

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


def write_args(model, args):
    graph_metadata = 'Model = "{}"\n{}'.format(model.__name__,
                                             toml.dumps(args.__dict__))

    tf.constant(graph_metadata, name="graph_metadata")

    f = open(args.write_args, 'w') if args.write_args else sys.stdout
    try:
        f.write(graph_metadata)
    finally:
        if f != sys.stdout:
            f.close()


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
    parser.add_argument(
        "--write_args",
        type=str,
        help="write the arguments to a file",
        default=None)
    return parser
