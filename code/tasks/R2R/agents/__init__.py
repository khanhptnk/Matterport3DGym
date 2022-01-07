from .nav_PREVALENT import NavAgentPREVALENT

def load(config):

    cls_name = config.nav_agent.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
