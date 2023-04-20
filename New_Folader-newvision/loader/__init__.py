import json


from loader.LF_dataset_loader import LFdatasetLoader
from loader.LFSyn_dataset_loader import LFSyndatasetLoader
def get_loader(name):
    """get_loader

    :param name:
    """
    return {"LF_dataset":LFdatasetLoader,"LFSyn_dataset":LFSyndatasetLoader,}[name]
  
