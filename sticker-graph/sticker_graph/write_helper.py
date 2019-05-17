import argparse
import tensorflow as tf
import toml

from sticker_graph.config import DefaultConfig


def read_shapes(args):
    with open(args.shape_file) as shapesfile:
        shapes = toml.loads(shapesfile.read())
    return shapes


def create_graph(config, model, args):
    shapes = read_shapes(args)
    graph_filename = args.output_graph_file

    gpuopts = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    tfconfig = tf.ConfigProto(gpu_options=gpuopts)

    with tf.Graph().as_default(), tf.Session(config=tfconfig) as session:
        with tf.variable_scope("model", reuse=None):
            m = model(config=config, shapes=shapes)

        init = tf.variables_initializer(tf.global_variables(), name='init')

        saver = tf.train.Saver(tf.global_variables())

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


def parse_common_config(args):
    config = DefaultConfig()
    config.crf = args.crf
    config.top_k = args.top_k
    return config
