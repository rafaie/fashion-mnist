"""
main.py: the base file to call the related class to train the model

"""


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
        default='./mnist/',
        help='directory where MNIST is located')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_class', default=BEST_MODEL)
    args = parser.parse_args()

    # load model class by class file name
    ModelClass = my_import(args.model_class + ".model")

    