import os
import pprint
import argparse
from datetime import datetime

from util.get_log import init_log
from util.get_dataset import build_dataset
from util.get_model import build_model

from engines.engine import enginer
from utils import *

def get_parser():
    parser = argparse.ArgumentParser()

    #model config
    # parser.add_argument("--cfg", required=True, help='select the config file')
    # parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/cot.yml')
    # parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/canet.yml')
    # parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/scen.yml')
    # parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/compcos.yml')
    # parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/prolt.yml') ## TODO: fix it
    # parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/ivr.yml')
    # parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/mit-states.yml')
    parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/scen_test.yml')

    parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
    parser.add_argument("--seed", help="seed value", default=0, type=int)
    parser.add_argument("--open_world", help="evaluate on open world setup", default= False)
    cfg = parser.parse_args()
    return cfg



def main():
    config = get_parser()
    log_time = datetime.now()
    load_args(config.cfg, config)
    print(config)

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