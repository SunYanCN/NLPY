import argparse
import numpy as np

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def init_gpu():
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    KTF.set_session(session)

def create_class_weight(labels):
    total = np.sum(labels) * 1.0
    ratio = labels.shape[0] / total
    class_weight = dict()
    class_weight[0] = 1 - ratio
    class_weight[1] = ratio
    return class_weight