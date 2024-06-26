import os
from models import *

def build_model(config, dataset):
    type = config.model_type

    if type.upper() == "DFSP":
        allattrs = dataset.attrs
        allobj = dataset.objs
        classes = [cla.replace(".", " ").lower() for cla in allobj]
        attributes = [attr.replace(".", " ").lower() for attr in allattrs]
        offset = len(attributes)
        model = DFSP(config, attributes=attributes, classes=classes, offset=offset)
    elif type.upper() == "COT":
        model = COT(dataset, config)
    elif type.upper() == "CANET":
        config.main_root = os.path.dirname(os.path.dirname(__file__))
        image_extractor = None
        if config.update_image_features:
            print('CANet Learnable image_embeddings')
            image_extractor = get_image_extractor(arch=config.image_extractor, pretrained=True)
            if not config.extract_feature_vectors:
                import torch.nn as nn
                image_extractor = nn.Sequential(*list(image_extractor.children())[:-1])
            image_extractor = image_extractor.to(config.device)
        canet = CANet(dataset, config).to(config.device)
        model = [image_extractor, canet]
    elif type.upper() == "SCEN":
        image_extractor = None
        if config.update_image_features:
            print('SCEN Learnable image_embeddings')
            image_extractor = get_image_extractor(arch =config.image_extractor, pretrained=True)
        scen = SCEN(dataset, config)
        scen.is_open=False
        model = [image_extractor, scen]
    elif type.upper() == "IVR":
        image_extractor = None
        if config.update_image_features:
            print('IVR Learnable image_embeddings')
            image_extractor = get_image_extractor(arch =config.image_extractor, pretrained=True)
        ivr = IVR(dataset, config)
        model = [image_extractor, ivr]
    elif type.upper() == "PROLT":
        image_extractor = image_decoupler = None
        if config.update_image_features:
            image_extractor = get_image_extractor(arch=config.image_extractor, pretrained=True)
            image_decoupler = get_image_extractor(arch=config.image_extractor, pretrained=True)
        prolt = CZSL(dataset, config)
        prolt.is_open = config.is_open
        model = [image_extractor, prolt, image_decoupler]
    elif type.upper() == "COMPCOS":
        image_extractor = None
        if config.update_image_features:
            print('SCEN Learnable image_embeddings')
            image_extractor = get_image_extractor(arch =config.image_extractor, pretrained=True)
        compcos = CompCos(dataset, config)
        model = [image_extractor, compcos]
    else:
        raise NotImplementedError('This method is not support now.')
    return model