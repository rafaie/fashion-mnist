"""
main.py: the base file to call the related class to train the model

"""

from util import get_data, get_dataset
from base.model import BaseModel, BaseConfig
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

def get_config():
    config = BaseConfig('fashion_mnist')

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify MNIST images.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/',
        help='directory where MNIST is located')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_class', default=BEST_MODEL)
    args = parser.parse_args()

    # set Logging 
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create and update configuration 
    config = get_config()
    config.n_epoch_train = 1
    config.n_epoch_eval = 1
    config.max_steps = args.max_epochs

    # Create datasets
    ds_train, ds_valid, ds_test, ds_pred = get_dataset(args.data_dir)
     

    # load model class by class file name
    ModelClass = my_import(args.model_class + ".model")
    model = ModelClass(config)

    n_epoch = 100

    if model.get_global_step() < 250:
        for i in range(1, n_epoch+1):

            model.train(ds_train)

            if i % 10 == 0:
                model.evaluate(ds_test)
    else:
        model.evaluate(ds_test)

    ret = model.predict(ds_pred)

    for i in ret:
        print(i)

