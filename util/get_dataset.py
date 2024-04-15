from datasets import *

def build_dataset(config, split, **args):

    if config.model_type.upper() == "COT":
        train_dataset = COTDataset(config, phase='train', split=config.splitname)
        val_dataset = COTDataset(config, phase='val', split=config.splitname)
        test_dataset = COTDataset(config, phase='test', split=config.splitname)
    elif config.model_type.upper() == "CANET":
        train_dataset = CANetDataset(config, 'train', config.splitname)
        val_dataset = CANetDataset(config, config.test_set, config.splitname)
        test_dataset = CANetDataset(config, config.test_set, config.splitname)
    elif config.model_type.upper() == "SCEN":
        train_dataset = SCENDataset(config, 'train', config.splitname)
        val_dataset = SCENDataset(config, config.test_set, config.splitname)
        test_dataset = SCENDataset(config, config.test_set, config.splitname)
    elif config.model_type.upper() == "IVR":
        train_dataset = IVRDataset(config, 'train', config.splitname)
        val_dataset = IVRDataset(config, config.test_set, config.splitname)
        test_dataset = IVRDataset(config, config.test_set, config.splitname)
    elif config.model_type.upper() == "PROLT":
        train_dataset = PROLTDataset(config, 'train', config.splitname)
        val_dataset = PROLTDataset(config, config.test_set, config.splitname)
        test_dataset = PROLTDataset(config, config.test_set, config.splitname)
    elif config.model_type.upper() == "COMPCOS":
        train_dataset = CompCosDataset(config, 'train', config.splitname)
        val_dataset = CompCosDataset(config, config.test_set, config.splitname)
        test_dataset = CompCosDataset(config, config.test_set, config.splitname)
    else:
        train_dataset = CompositionDataset(config.dataset_path, phase='train', split=split)
        val_dataset = CompositionDataset(config.dataset_path, phase='val', split=split)
        test_dataset = CompositionDataset(config.dataset_path, phase='test', split=split)

    return train_dataset, val_dataset, test_dataset