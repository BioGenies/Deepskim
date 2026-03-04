import os
import random
import numpy as np
from typing import Any, Dict
from abc import ABC, abstractmethod

import torch
from transformers import set_seed, Trainer
from datasets import Dataset


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

# ---------- Strategy / Builder ----------
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Train a model on a single HuggingFace Dataset and return objects needed for inference.
        Returns a dict containing at least: {'model': model, 'tokenizer': tokenizer, 'trainer': trainer}
        """
        pass

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