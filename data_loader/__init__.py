from .cityscapes import CitySegmentation
from .sim_out_type1 import sim_out_type1

datasets = {
    'citys': CitySegmentation,
    'sim_out_1' : sim_out_type1,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
