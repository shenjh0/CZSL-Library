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
    else:
        raise NotImplementedError('This method is not support now.')
    
    return model