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

    gpuopts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    tfconfig = tf.compat.v1.ConfigProto(gpu_options=gpuopts)

    with tf.Graph().as_default(), tf.compat.v1.Session(config=tfconfig) as session:
        write_args(model, args)

        logdir = tf.compat.v1.placeholder(
            shape=[], name="logdir", dtype=tf.string)
        summary_writer = sticker_graph.vendored._create_file_writer_generic_type(
            logdir)

        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            with tf.compat.v1.variable_scope("model", reuse=None):
                model(args=args, shapes=shapes)

            tf.group(tf.contrib.summary.summary_writer_initializer_op(),
                     name="summary_init")
            tf.compat.v1.variables_initializer(
                tf.compat.v1.global_variables(), name='init')

            tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            serialized_graph = session.graph_def.SerializeToString()
            serialized_graph_tensor = tf.convert_to_tensor(serialized_graph)
            tf.contrib.summary.graph(
                serialized_graph_tensor, 0, name='graph_write')
            tf.io.write_graph(
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
        "--byte_embed_size",
        type=int,
        help="size of character embeddings",
        default=25)
    parser.add_argument(
        "--crf",
        help="use CRF layer for classification",
        action="store_true")
    parser.add_argument(
        "--subword_gru",
        help="use GRU RNN cells in the character RNN",
        action="store_true")
    parser.add_argument(
        "--subword_hidden_size",
        type=int,
        help="character RNN hidden size per direction",
        default=25)
    parser.add_argument(
        "--subword_keep_prob",
        type=float,
        help="character RNN dropout keep probability",
        default=0.6)
    parser.add_argument(
        "--subword_layers",
        type=int,
        help="character RNN hidden layers",
        default=2)
    parser.add_argument(
        "--subword_len",
        type=int,
        help="number of characters in character-based representations",
        default=40)
    parser.add_argument(
        "--subword_residual",
        action='store_true',
        help="use character RNN residual skip connections"
    )
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
