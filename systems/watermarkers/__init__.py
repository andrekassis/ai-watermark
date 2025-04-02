import yaml
from watermarkers.networks import *


def init_watermarker(name, batch_size=64, device="cuda"):
    sys_map = {
        "yu1": Yu1,
        "yu2": Yu2,
        "ptw": PtW,
        "hidden": HiDDeN,
        "stega": Stega,
        "treering": TREERING,
        "stablesignature": StableSignature,
        "prc": Prc,
        "gs": Gs,
        "vine": Vine,
    }
    assert name in list(sys_map.keys())
    constructor = sys_map[name]
    conf_path = "systems/configs/" + name + ".yaml"
    with open(conf_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    return constructor(batch_size=batch_size, device=device, **conf)
