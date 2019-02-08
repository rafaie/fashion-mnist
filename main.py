"""
main.py: the base file to call the related class to train the model

"""

from util import get_data, get_dataset
from base_config import BaseConfig
import tensorflow as tf
import argparse
import os


BEST_MODEL="model"

def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify MNIST images.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/',
        help='directory where MNIST is located')
    parser.add_argument('--model_class', default=BEST_MODEL)
    args = parser.parse_args()

    # set Logging 
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create datasets
    ds_train, ds_valid, ds_test, ds_pred = get_dataset(args.data_dir, test_size = 0.2, valid_size = 0.001)

    # load model class by class file name
    ModelClass = my_import(args.model_class + ".model")
    model = ModelClass()

    model.train_and_evaluate(ds_train, ds_test)

