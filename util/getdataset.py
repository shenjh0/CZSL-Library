from dataset import CompositionDataset, COTDataset


def build_dataset(config, split, **args):

    if config.model_type.upper() == "COT":
        train_dataset = COTDataset(phase='train', split=split, cfg=config)
        val_dataset = COTDataset(phase='val', split=split, cfg=config)
        test_dataset = COTDataset(phase='test', split=split, cfg=config)
    else:
        train_dataset = CompositionDataset(config.dataset_path,
                                        phase='train',
                                        split=split)
        val_dataset = CompositionDataset(config.dataset_path,
                                        phase='val',
                                        split=split)

        test_dataset = CompositionDataset(config.dataset_path,
                                        phase='test',
                                        split=split)

    return train_dataset, val_dataset, test_dataset