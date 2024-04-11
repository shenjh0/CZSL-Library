import argparse
import os
import pickle
import pprint
from datetime import datetime

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from parameters import parser, YML_PATH
from loss import loss_calu
import test as test

from dataset import CompositionDataset
from util.getdataset import build_dataset
from util.get_model import build_model
from engine import train, evaluate
from models import DFSP, DRPT
from utils import *


def main():
    config = parser.parse_args()
    log_time = datetime.now()
    os.makedirs(os.path.join("logs", config.dataset), exist_ok=True)
    set_log("logs/" + config.dataset + '/' + str(log_time) + ".log")

    load_args(config.cfg, config)
    logging.warning(config.log_id)
    logging.info(config)
    print(config)
    # set the seed value

    set_seed(config.seed)

    train_dataset, val_dataset, test_dataset = build_dataset(config, 'compositional-split-natural')

    # TODO: pdrpt needs these
    if False:
        ent_attr, ent_obj = train_dataset.ent_attr, train_dataset.ent_obj

    # model = DRPT(config, attributes=attributes, classes=classes, offset=offset, ent_attr=ent_attr, ent_obj=ent_obj).cuda()
    model = build_model(config, train_dataset)
    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    os.makedirs(config.save_path, exist_ok=True)

    train(model, optimizer, config, train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    print("done!")


if __name__ == "__main__":
    main()