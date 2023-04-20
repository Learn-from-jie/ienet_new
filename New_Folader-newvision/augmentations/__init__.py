import logging

from augmentations.augmentations import (
    Compose,
    RandomFlip,
    RandomCrop,
    RandomResize,
    PhotoMetricDistortion
)

logger = logging.getLogger("ptsemseg")

key2aug = {
    "colorjtr": PhotoMetricDistortion,
    "resize": RandomResize,
    "prob": RandomFlip,
    "crop": RandomCrop
}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None
    

    augmentations = []
    for aug_key, aug_param in aug_dict.items():     
        augmentations.append(key2aug[aug_key](aug_param))   
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)  
