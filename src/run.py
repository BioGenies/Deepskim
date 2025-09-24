import torch
from transformers import set_seed, Trainer
from copy import deepcopy
import wandb
from IPython.display import clear_output

from data_prep import prepare_dataset
from model_building import BERTClassificationStrategy


DEFAULT_CONFIG = {
    "seed": 42,
    "device": None,  # "cuda" or "cpu"
    "data": {
        "file_path": "amyloid-02-07-2025.csv",
        "target": "rejection",
        "split_strategy": "train_test",
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "max_length": 20,
        "keywords": ["amyloid"],
        "special_tokens": ["[KEY]"],
        "seed": 42,
    },
    "model": {
        "model_name": "bert-base-uncased",
        "num_labels": 2,
        "special_tokens": [],
        "unfreeze_last_k_layers": 6,
        "change_classifier": False,
    },
    "training": {
        "hparams": {
            "learning_rate": 2e-5,
            "num_train_epochs": 10,
            "per_device_train_batch_size": 16,
            "weight_decay": 0.01,
            "classifier_hidden_dim": 512,
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
    wandb.init(project="Amyloid2", name=run_name)
    train, test = prepare_dataset(**config["data"])

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
    test_metrics = eval_trainer.evaluate(
        eval_dataset=test["test"], metric_key_prefix="test"
    )

    metrics_to_log = {**train_metrics, **val_metrics, **test_metrics, **wandb_config}
    wandb.log(metrics_to_log)

    return artifacts


def cv_main(config):
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
        "title_abstract_journal": True,
    }
    run_name = f"{config['model']['model_name']}_lr{config['training']['hparams']['learning_rate']}_unfreeze{wandb_config['unfreeze_last_k_layers']}__ml{wandb_config['seq_max_length']}__seed{config['seed']}"
    train, test = prepare_dataset(**config["data"])

    for i, dataset in enumerate(train):
        wandb.init(project="Amyloid", name="title_" + run_name + f"_cv{i}")
        wandb_config["cv"] = i

        strategy = BERTClassificationStrategy(
            model_name=config["model"]["model_name"],
            num_labels=config["model"]["num_labels"],
            hparams=config["training"]["hparams"],
            special_tokens=config["model"].get("special_tokens", []),
            unfreeze_last_k_layers=config["model"].get("unfreeze_last_k_layers", 6),
            device=device,
            change_classifier=config["model"]["change_classifier"],
        )

        artifacts = strategy.build_and_train_model(dataset)

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
            eval_dataset=dataset["train"], metric_key_prefix="best_train"
        )
        val_metrics = eval_trainer.evaluate(
            eval_dataset=dataset["test"], metric_key_prefix="best_val"
        )
        test_metrics = eval_trainer.evaluate(
            eval_dataset=test["test"], metric_key_prefix="test"
        )

        metrics_to_log = {
            **train_metrics,
            **val_metrics,
            **test_metrics,
            **wandb_config,
        }
        wandb.log(metrics_to_log)
        wandb.finish()
        clear_output()

    return artifacts


if __name__ == "__main__":
    main()
