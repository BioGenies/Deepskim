import os
import random
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import Dataset
from torch import nn

print("Imported model building :)")


# ---------- Reproducibility helper ----------
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # transformers helper
    set_seed(seed)
    # Torch deterministic flags (may impact performance / availability)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # fallback for older torch versions
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------- Model configurator with safer attribute access ----------
class ModelConfigurator:
    def __init__(
        self,
        special_tokens: list[str],
        unfreeze_last_k_layers: int = 6,
        change_classifier: bool = False,
    ):
        self.special_tokens = special_tokens
        self.unfreeze_last_k = unfreeze_last_k_layers
        self.change_classifier = change_classifier

    def add_special_tokens(self, tokenizer, model):
        logging.info("Adding special tokens and resizing model embeddings.")
        tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        model.resize_token_embeddings(len(tokenizer))

    def _get_base_model(self, model):
        # handle variations like model.bert, model.roberta, model.base_model, etc.
        if hasattr(model, "base_model"):
            return model.base_model
        for attr in ["bert", "roberta", "distilbert", "electra"]:
            if hasattr(model, attr):
                return getattr(model, attr)
        return None

    def freeze_layers(self, model):
        logging.info(
            "Freezing most model parameters, unfreezing classifier, embeddings and last K encoder layers (if present)."
        )
        # freeze all first
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze classifier head if exists
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

        base = self._get_base_model(model)
        # unfreeze pooler if present
        if base is not None and hasattr(base, "pooler"):
            try:
                for param in base.pooler.parameters():
                    param.requires_grad = True
            except Exception:
                pass

        # try to unfreeze embeddings
        if base is not None and hasattr(base, "embeddings"):
            for param in base.embeddings.parameters():
                param.requires_grad = True

        # unfreeze last K encoder layers (if encoder exists)
        if (
            base is not None
            and hasattr(base, "encoder")
            and hasattr(base.encoder, "layer")
        ):
            encoder_layers = list(base.encoder.layer)
            for layer in encoder_layers[-self.unfreeze_last_k :]:
                for param in layer.parameters():
                    param.requires_grad = True

    def add_two_layer_classifier(
        self,
        model,
        num_classes: int = 2,
        hidden_size: int = 512,
        dropout_rate: float = 0.1,
    ):
        """
        Replace the existing classifier with a two-layer MLP classifier.

        Args:
            model: The transformer model (BERT, RoBERTa, etc.)
            num_classes: Number of output classes
            hidden_size: Size of the hidden layer (default: 512)
            dropout_rate: Dropout rate between layers (default: 0.1)
        """
        logging.info(
            f"Adding two-layer classifier with hidden_size={hidden_size}, num_classes={num_classes}"
        )

        # Create new two-layer classifier
        new_classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

        # Replace the classifier
        if hasattr(model, "classifier"):
            # For BERT-style models
            if isinstance(model.classifier, nn.Linear):
                # Simple linear classifier
                model.classifier = new_classifier
            else:
                # Complex classifier head (like RoBERTa)
                model.classifier = new_classifier
        else:
            raise ValueError("Model does not have a 'classifier' attribute")

        logging.info("Two-layer classifier successfully added")
        return model


# ---------- Custom Trainer ----------
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss


# ---------- Metrics (safe for binary; change average for multiclass) ----------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": float((preds == labels).mean()),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
    }


# ---------- Strategy / Builder ----------
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Train a model on a single HuggingFace Dataset and return objects needed for inference.
        Returns a dict containing at least: {'model': model, 'tokenizer': tokenizer, 'trainer': trainer}
        """
        pass


# Concrete Strategy for fine-tuning BERT-based classifier on one dataset
class BERTClassificationStrategy(ModelBuildingStrategy):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        hparams: Dict[str, Any],
        special_tokens: list[str],
        unfreeze_last_k_layers: int,
        device: torch.device = None,
        change_classifier: bool = False,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.hparams = hparams
        self.device = device
        self.configurator = ModelConfigurator(
            special_tokens, unfreeze_last_k_layers, change_classifier
        )
        """
        hparams: 
            {
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 16,
                "num_train_epochs": 3,
                "weight_decay": 0.05,
                "warmup_ratio": 0.1
            }
        """

    def build_and_train_model(self, dataset: Dataset) -> Dict[str, Any]:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        model.to(self.device)

        # Configure tokenizer/model
        self.configurator.add_special_tokens(tokenizer, model)
        self.configurator.freeze_layers(model)
        if self.configurator.change_classifier:
            self.configurator.add_two_layer_classifier(
                model, self.num_labels, self.hparams["classifier_hidden_dim"]
            )

        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        temp_dir = "./bert"
        # with tempfile.TemporaryDirectory() as temp_dir:
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=temp_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.hparams["per_device_train_batch_size"],
            per_device_eval_batch_size=16,
            num_train_epochs=self.hparams["num_train_epochs"],
            learning_rate=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            warmup_ratio=self.hparams["warmup_ratio"],
            logging_steps=10,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=self.device.type == "cuda",
            report_to=["wandb"],
            seed=42,
        )

        # Compute class weights for CV train split
        labels = np.array(dataset["train"]["labels"])
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(labels), y=labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        # Split dataset into train and validation set
        trainer = CustomTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
        )

        logging.info("Starting training.")
        trainer.train()

        return {"model": model, "tokenizer": tokenizer, "trainer": trainer}


# Context Class to use strategy
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, dataset: Dataset) -> Dict[str, Any]:
        logging.info("Building and training the model on single dataset.")
        return self._strategy.build_and_train_model(dataset)
