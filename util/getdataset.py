from dataset import CompositionDataset, COTDataset, CANetDataset


def build_dataset(config, split, **args):

    if config.model_type.upper() == "COT":
        train_dataset = COTDataset(phase='train', split=config.splitname, cfg=config)
        val_dataset = COTDataset(phase='val', split=config.splitname, cfg=config)
        test_dataset = COTDataset(phase='test', split=config.splitname, cfg=config)
    elif config.model_type.upper() == "CANET":
        train_dataset = CANetDataset(
        config, config.dataset_path, 'train', split=config.splitname,
        model =config.image_extractor, update_image_features = config.update_image_features,
        train_only= config.train_only)
        val_dataset = CANetDataset(
        config, config.dataset_path, config.test_set, split=config.splitname, 
        model =config.image_extractor, update_image_features = config.update_image_features)
        test_dataset = CANetDataset(
        config, config.dataset_path, config.test_set, split=config.splitname, 
        model =config.image_extractor, update_image_features = config.update_image_features)
    else:
        train_dataset = CompositionDataset(config.dataset_path, phase='train', split=split)
        val_dataset = CompositionDataset(config.dataset_path, phase='val', split=split)
        test_dataset = CompositionDataset(config.dataset_path, phase='test', split=split)

    return train_dataset, val_dataset, test_dataset