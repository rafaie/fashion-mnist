"""
main.py: the base file to call the related class to train the model

"""

from util import get_data, get_dataset
from model import FashionMnist
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify MNIST images.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/',
        help='directory where MNIST is located')
    parser.add_argument(
        '--model_id',
        type=str,
        default=0,
        help='rum the model id')
    parser.add_argument('--batch_size', type=int, default=200)
    args = parser.parse_args()

    m = FashionMnist(args.data_dir, args.batch_size)
    m.show_dataset_size()
    m.train(model_id=args.model_id)