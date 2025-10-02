import torch
from transformers import set_seed, Trainer
from copy import deepcopy
import wandb

from .prepare_data import prepare_dataset
from model_building import BERTClassificationStrategy


model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

DEFAULT_CONFIG = {
    "seed": 1,
    "device": "cuda",  # "cuda" or "cpu"
    "wandb_project_name": "Amyloid-test",
    "data": {
        "train_file_path": "train.csv",
        "test_file_path": "test.csv",
        "target": "Rejection?",
        "test_size": 0.15,
        "model_name": model_name,
        "max_length": 380,
        "keywords": [
            "aggregates",
            "amyloid",
            "scfv",
            "hiapp",
            "mab",
            "ttr",
            "donanemab",
            "aggregation",
        ],
        "special_tokens": ["[KEY]", "[J_END]", "[T_END]"],
        "seed": 1,
    },
    "model": {
        "model_name": model_name,
        "num_labels": 2,
        "special_tokens": ["[KEY]", "[J_END]", "[T_END]"],
        "unfreeze_last_k_layers": 12,
        "change_classifier": False,
    },
    "training": {
        "hparams": {
            "learning_rate": 3e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 16,
            "weight_decay": 0.1,
            "classifier_hidden_dim": 512,
            "warmup_ratio": 0.1,
        }
    },
}


def main(config):
    set_seed(config.get("seed", 42))

    if config.get("device"):
        device = torch.device(config["device"])  # "cuda" or "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    wandb_config = {
        "unfreeze_last_k_layers": config["model"]["unfreeze_last_k_layers"],
        "seq_max_length": config["data"]["max_length"],
        "change_classifier": config["model"]["change_classifier"],
        "classifier_hidden_dim": config["training"]["hparams"]["classifier_hidden_dim"],
    }
    run_name = f"{config['model']['model_name']}_lr{config['training']['hparams']['learning_rate']}_unfreeze{wandb_config['unfreeze_last_k_layers']}__ml{wandb_config['seq_max_length']}__seed{config['seed']}"
    wandb.init(project=config["wandb_project_name"], name=run_name)
    train, hf_test = prepare_dataset(**config["data"])

    strategy = BERTClassificationStrategy(
        model_name=config["model"]["model_name"],
        num_labels=config["model"]["num_labels"],
        hparams=config["training"]["hparams"],
        special_tokens=config["model"].get("special_tokens", []),
        unfreeze_last_k_layers=config["model"].get("unfreeze_last_k_layers", 6),
        device=device,
        change_classifier=config["model"]["change_classifier"],
    )

    artifacts = strategy.build_and_train_model(train)

    trainer = artifacts["trainer"]
    eval_args = deepcopy(trainer.args)
    eval_args.report_to = []
    eval_args.eval_strategy = "no"

    eval_trainer = Trainer(
        model=trainer.model,
        args=eval_args,
        compute_metrics=trainer.compute_metrics,
    )

    train_metrics = eval_trainer.evaluate(
        eval_dataset=train["train"], metric_key_prefix="best_train"
    )
    val_metrics = eval_trainer.evaluate(
        eval_dataset=train["test"], metric_key_prefix="best_val"
    )

    metrics_to_log = {**train_metrics, **val_metrics, **wandb_config}
    wandb.log(metrics_to_log)
    wandb.finish()

    return artifacts, hf_test
