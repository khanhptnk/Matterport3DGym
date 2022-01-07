from .env import *

def load(config):
    cls_name = config.environment.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such teacher: {}".format(cls_name))
