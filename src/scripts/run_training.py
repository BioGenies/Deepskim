import argparse
import os
import yaml
from transformers import set_seed
from deployment import train_model_bert, train_model_qlora

os.environ["WANDB_PROJECT"] = "Amyloid-test"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def main(args):
    config = args.config
    if config is None:
        config = "configs/default.yaml"

    config = yaml.safe_load(open(config, "r"))

    set_seed(config.get("seed", 42))
    if "peft" in config and config["peft"]["method"] == "lora":
        print("Training with QLoRA...")
        train_model_qlora(config)
    else:
        print("Training with BERT classification...")
        artifacts, hf_test = train_model_bert(config)
    "Training completed."
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args)
