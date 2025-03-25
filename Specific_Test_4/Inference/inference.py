import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from utils import infer, download_weights

state_dict = download_weights()

args = {
    "config_path": "config.yaml",
}

infer(args, state_dict)
