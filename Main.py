import argparse
import logging as lgg
import time
import warnings

import torch
import transformers
from torch import nn

import yh_model
import utils
from config import Config
from train import train_attention
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')
# def model():


def setlogging(level, filename):
    for handler in lgg.root.handlers[:]:
        lgg.root.removeHandler(handler)
    lgg.basicConfig(filename=filename, level=level, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M',
                    filemode='w')

    logc = lgg.StreamHandler()
    logc.setLevel(level=lgg.DEBUG)
    logc.setFormatter(lgg.Formatter('%(message)s'))
    lgg.getLogger().addHandler(logc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yh_model', help='choose a model')
    parser.add_argument('--cuda', type=int, default=0, choices=[0, 1], help='choose a cuda: 0 or 1')
    parser.add_argument('--tune', type=int, default=1, choices=[1, 0], help='fine tune or not: 0 or 1')
    parser.add_argument('--base', type=str, default='roberta', choices=['roberta'], help='roberta model as encoder')
    parser.add_argument('--config', type=str, default='config/pdtb2_4.json', help='choose a model')
    args = parser.parse_args()

    model_name = args.model  # bert
    # x = import_module(model_name)

    config = Config(args, args.cuda, bool(args.tune), args.base)

    setlogging(lgg.DEBUG, config.log)

    hyper_parameters = config.__dict__.copy()
    lgg.info(hyper_parameters)
    start_time = time.time()
    train, dev, test = utils.build_dataset(config)
    train_iter = utils.build_iterator(train, config)
    dev_iter = utils.build_iterator(dev, config)
    test_iter = utils.build_iterator(test, config)

    time_dif = utils.get_time_dif(start_time)
    lgg.info("Time usage: {}".format(time_dif))

    # train
    model = yh_model.model_all(config.embed_dim, config).to(device='cuda')
    criterion = nn.CrossEntropyLoss().to(device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_attention(train_iter, test_iter, dev_iter, model, criterion, config)
