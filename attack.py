import os
import random
import argparse
import yaml
import warnings
import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch

from modules.logs import get_logger

logger = get_logger()

from modules.attack import *


def dict_to_str(d, prefix=""):
    return f"{', '.join([ (f'{k}{prefix}: {v}') if not isinstance(v, dict) else dict_to_str(v, '' if k != 'detection rates' else ' detection rate') for k, v in d.items() ])}"


def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument(
        "--attack",
        "-a",
        choices=[
            "ID",
            "UnMarker",
            "DiffusionAttack",
            "VAEAttack",
            "Crop",
            "JPEG",
            "SuperResolution",
            "Quantize",
            "Blur",
            "GuidedBlur",
            "Noise",
        ],
        required=True,
    )
    parser.add_argument(
        "--evaluator",
        "-e",
        choices=[
            "Yu1",
            "Yu2",
            "PTW",
            "HiDDeN",
            "TreeRing",
            "StegaStamp",
            "StableSignature",
            "Prc",
            "Gs",
            "Vine",
        ],
        required=True,
    )
    parser.add_argument("--total_imgs", "-t", type=int, default=100)
    parser.add_argument("--seed", "-s", type=int, default=1234)
    parser.add_argument("--device", "-d", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    attack_name = args.attack if args.attack != "UnMarker" else "UnMark"
    attack_name = attack_name if attack_name != "VAEAttack" else "VAE"
    sysname = args.evaluator if args.evaluator != "StegaStamp" else "Stega"

    with open(os.path.join("attack_configs", args.evaluator + ".yaml")) as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    input_dir = conf.get("input_dir")
    conf = conf.get(args.attack, {})

    output_dir = args.output_dir
    if os.path.exists(output_dir):
        logger.error("Output directory exists. Aborting!")
        exit(1)

    os.makedirs(output_dir)

    attacker = eval(attack_name)(
        sysname.lower(),
        input_dir=input_dir,
        output_dir=output_dir,
        device=args.device,
        **conf,
    )
    detection_threshold = attacker.evaluator.acceptance_thresh

    logger.info(
        f"Evaluating scheme: {args.evaluator} at detection threshold {detection_threshold} "
        f"against {args.attack}.\n"
    )

    result = attacker(args.total_imgs)

    result["attack"] = args.attack
    result["scheme"] = args.evaluator
    result["detection threshold"] = detection_threshold
    with open(os.path.join(attacker.output_dir, "aggregated_results.yaml"), "w") as f:
        yaml.dump(result, f, default_flow_style=False)
    result["output dir"] = attacker.output_dir
    logger.info(dict_to_str(result))
