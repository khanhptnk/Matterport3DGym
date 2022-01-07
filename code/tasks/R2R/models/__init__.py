from .model_VLNBERT import VLNBERT

def load(config):
    cls_name = config.nav_agent.model.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
