import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

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
)
from datasets import Dataset

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Utility class for model configuration
class ModelConfigurator:
    def __init__(
        self,
        special_tokens: list[str],
        freeze_last_k_layers: int = 6,
        device: torch.device = None,
    ):
        self.special_tokens = special_tokens
        self.freeze_last_k = freeze_last_k_layers
        self.device = device

    def add_special_tokens(self, tokenizer, model):
        logging.info("Adding special tokens and resizing model embeddings.")
        tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        model.resize_token_embeddings(len(tokenizer))

    def freeze_layers(self, model):
        logging.info(
            "Freezing model layers, except pooler, classifier, last K encoder layers, and embeddings."
        )
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.base_model.pooler.parameters():
            param.requires_grad = True
        encoder_layers = list(model.base_model.encoder.layer)
        for layer in encoder_layers[-self.freeze_last_k :]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = True


# Custom Trainer to incorporate class weights
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": (preds == labels).mean(),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


# Abstract Base Class for single-dataset training
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
        special_tokens: list[str],
        epochs: int = 5,
        freeze_last_k_layers: int = 6,
        device: torch.device = None,
        test_split_ratio: float = 0.1,
    ):
        self.model_name = model_name
        self.epochs = epochs
        self.device = device
        self.configurator = ModelConfigurator(
            special_tokens, freeze_last_k_layers, device
        )
        self.test_ratio = test_split_ratio

    def build_and_train_model(self, dataset: Dataset) -> Dict[str, Any]:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        model.to(self.device)

        # Configure tokenizer/model
        self.configurator.add_special_tokens(tokenizer, model)
        self.configurator.freeze_layers(model)

        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir="./bert_finetuned",
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=self.epochs,
            learning_rate=1e-5,
            weight_decay=0.05,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=False,
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


# # Example usage:
# if __name__ == "__main__":
#     # Detect device
#     device = torch.device(
#         "mps"
#         if torch.backends.mps.is_available()
#         else "cuda"
#         if torch.cuda.is_available()
#         else "cpu"
#     )
#     logging.info(f"Using device: {device}")

#     strategy = BERTClassificationStrategy(
#         model_name="bert-base-uncased",
#         special_tokens=["[KEY]"],
#         epochs=3,
#         freeze_last_k_layers=4,
#         device=device,
#     )
#     builder = ModelBuilder(strategy)
#     # Assume 'dataset' is a HuggingFace Dataset with column 'text' and 'label'
#     artifacts = builder.build_model(dataset)
#     # artifacts['model'], artifacts['tokenizer'], artifacts['trainer'] can be used for inference
