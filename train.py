import argparse
import os

import pprint
from datetime import datetime

from parameters import parser

from util.get_dataset import build_dataset
from util.get_model import build_model
from engines.engine import enginer
from utils import *
from util.get_log import init_log

def main():
    config = parser.parse_args()
    log_time = datetime.now()
    load_args(config.cfg, config)
    print(config)
    # set the seed value

    config.logdir = os.path.join("logs/", config.model_type, config.dataset, str(log_time).replace(' ', '_'))
    logpath = os.path.join("logs/", config.model_type, config.dataset, str(log_time).replace(' ', '_'), 
                           "log.log")
    logger = init_log('global', logpath, logging.INFO)
    logger.info('{}\n'.format(pprint.pformat({**vars(config)})))

    set_seed(config.seed)

    # train_dataset, val_dataset, test_dataset = build_dataset(config, 'compositional-split-manual')
    train_dataset, val_dataset, test_dataset = build_dataset(config, config.splitname)

    # TODO: pdrpt needs these
    if False:
        ent_attr, ent_obj = train_dataset.ent_attr, train_dataset.ent_obj

    # model = DRPT(config, attributes=attributes, classes=classes, offset=offset, ent_attr=ent_attr, ent_obj=ent_obj).cuda()
    model = build_model(config, train_dataset)
    if isinstance(model, list):
        model = [module.cuda() if module is not None else module for module in model]
    else:
        model = model.cuda()

    enginer(model, config, train_dataset, val_dataset, test_dataset, logger)


if __name__ == "__main__":
    main()