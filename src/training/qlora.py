import os
import random
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from functools import partial
import numpy as np

from datasets import Dataset, DatasetDict
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import precision_score, recall_score, f1_score

from .bert_model_building import compute_metrics
from utils.evaluation import (
    gather_yes_no_logprobs,
    convert_scores_to_probs,
    convert_probs_to_labels,
    percent_to_review_for_recall,
)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


y_true_list = []
y_pred_list = []
scores_list = []
explain_true_list = []
explain_pred_list = []


def compute_metrics(
    eval_preds, tokenizer, compute_result, shift=True, explainability=False
):
    global y_true_list, y_pred_list, scores_list, reason_true_list, reason_pred_list
    # Compute include/exclude (yes/no) metrics only
    logits = eval_preds.predictions  # [batch, seq_len,, vocab]
    labels = eval_preds.label_ids  # [batch, seq_len]

    scores = gather_yes_no_logprobs(logits, tokenizer)  # [batch, seq_len, 2]
    scores = convert_scores_to_probs(
        scores.view(-1, 2)
    )  # [batch*seq_len, 2] -> [batch*seq_len]
    scores = scores.view(labels.shape)  # [batch, seq_len]
    preds = logits.argmax(-1)  # [batch, seq_len]

    if shift:
        preds = preds[:, :-1]
        scores = scores[:, :-1]
        labels = labels[:, 1:]

    preds = preds.cpu()
    labels = labels.cpu()

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]

    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)
    scores_flat = scores.reshape(-1)

    mask = np.isin(labels_flat, [yes_id, no_id])
    # print(f"last pred tokens: {preds[:,-10:]}")
    # print(f"last label tokens: {labels[:,-10:]}")

    if mask.sum() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true_tokens = labels_flat[mask]
    y_pred_tokens = preds_flat[mask]
    scores_flat = scores_flat[mask]
    print(f"Predicted: {y_pred_tokens}, True: {y_true_tokens}")

    y_true = (y_true_tokens == yes_id).numpy().astype(int)
    y_pred = (y_pred_tokens == yes_id).numpy().astype(int)

    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    scores_list.append(scores_flat.cpu().numpy())

    if explainability:

        B, T, V = logits.shape
        pred_select = preds_flat[labels_flat != -100]
        pred = pred_select.view(B, -1)
        labels_select = labels_flat[labels_flat != -100]
        labels = labels_select.view(B, -1)

        explain_pred = pred[
            :, -1
        ]  # Selet only the last generated token (reason for exclusion or n if included)
        explain_true = labels[
            :, -1
        ]  # Selet only the last generated token (reason for exclusion or n if included)
        explain_true_list.append(explain_true.numpy())
        explain_pred_list.append(explain_pred.numpy())

    if compute_result:
        y_true = np.concatenate(y_true_list, 0)
        y_pred = np.concatenate(y_pred_list, 0)
        scores = np.concatenate(scores_list, 0)
        explain_true = np.concatenate(explain_true_list, 0)
        explain_pred = np.concatenate(explain_pred_list)
        metrics = {
            "accuracy": float((y_pred == y_true).mean()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "% to review for 95% recall": percent_to_review_for_recall(
                list(zip(y_pred, scores)), y_true, recall_target=0.95
            ),
            "explain_accuracy": float((explain_pred == explain_true).mean()),
        }
        y_true_list = []
        y_pred_list = []
        scores_list = []
        explain_true_list = []
        explain_pred_list = []
        return metrics


class QLora:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        # num_labels: int,
        lora_config: Dict[str, Any],
        sft_config: Dict[str, Any],
        train_dataset: Dataset,
        device: torch.device = torch.device("cpu"),
        eval_dataset: Dataset = None,
    ):
        self.model = model
        # self.num_labels = num_labels
        self.device = device
        self.lora_config = LoraConfig(**lora_config)
        self.sft_config = SFTConfig(**sft_config)
        compute_metrics_func = partial(
            compute_metrics, tokenizer=tokenizer, explainability=True
        )
        self.trainer = WeightedCEExplainSFTTrainer(
            self.model,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics_func,
            args=self.sft_config,
            peft_config=self.lora_config,
            eval_dataset=eval_dataset,
        )

    def train_model(self):
        assert self.trainer.train_dataset is not None, "Train dataset not set."
        self.trainer.train()


class WeightedCESFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # Define weights: [Weight for Exclude (0), Weight for Include (1)]
        # We give the positive class a weight of ~11.5
        # weights = torch.tensor([1.0, 11.5]).to(model.device)
        weights = torch.ones((logits.shape[-1],), device=model.device)
        weights[5081] = 11.5  # yes token
        weights[708] = 1  # no token
        # Flatten logits and labels for CrossEntropy
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        B, T, V = logits.shape
        loss = loss_fct(logits.view(B * T, V), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class WeightedCEExplainSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_decision = 2
        self.lambda_reason = 1
        self.lambda_struct = 0.1
        self._last_sub_losses: Dict[str, float] = {}

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        logs.update(self._last_sub_losses)
        if start_time is not None:
            super().log(logs, start_time=start_time)
        else:
            super().log(logs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # Select all generated tokens
        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)

        logits_select = logits_flat[labels_flat != -100]
        labels_select = labels_flat[labels_flat != -100]

        logits = logits_select.view(B, -1, V)
        labels = labels_select.view(B, -1)
        B, T, V = logits.shape  # Note shape change

        # Loss has two parts: yes/no classification (weighted) + reason selection

        # Inclusion decision: yes/no classification
        # Define weights: [Weight for Exclude (0), Weight for Include (1)]
        # We give the positive class a weight of ~11.5
        # weights = torch.tensor([1.0, 11.5]).to(model.device)
        weights = torch.ones((logits.shape[-1],), device=model.device)
        weights[5081] = 11.5  # yes token
        weights[708] = 1  # no token
        # Flatten logits and labels for CrossEntropy
        # loss_func_include = nn.CrossEntropyLoss(weight=weights) #TODO: Change to functional to avoid unnecessary
        decision_logits = logits[
            :, 0
        ]  # Selet only the first generated token (inclusion/exclusion logit)
        decision_labels = labels[:, 0]
        loss_include = F.cross_entropy(
            decision_logits.view(B, V), decision_labels.view(-1), weight=weights
        )

        # Reasoning decision: Select a decision for exclusion from the codebook
        reason_logits = logits[
            :, -1
        ]  # Selet only the last generated token (reason for exclusion or n if included)
        reason_labels = labels[
            :, -1
        ]  # Selet only the last generated token (reason for exclusion or n if included)

        loss_reason = F.cross_entropy(reason_logits.view(B, V), reason_labels.view(-1))

        # Add a small overall loss to enforce structure
        loss_struct = F.cross_entropy(logits.view(B * T, V), labels.view(B * T))
        loss = (
            self.lambda_decision * loss_include
            + self.lambda_reason * loss_reason
            + self.lambda_struct * loss_struct
        )

        self._last_sub_losses = {
            "loss_decision": loss_include.detach().item(),
            "loss_reason": loss_reason.detach().item(),
            "loss_struct": loss_struct.detach().item(),
        }

        return (loss, outputs) if return_outputs else loss


class FocalLossSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def binary_focal_from_logits(
        margin: torch.Tensor, y: torch.Tensor, alpha=0.25, gamma=2.0
    ):
        """
        margin: [B] = logit_yes - logit_no
        y: [B] in {0,1} where 1 means "yes"
        """
        y = y.float()
        p = torch.sigmoid(margin)

        # Stable log terms
        logp = F.logsigmoid(margin)  # log(sigmoid(m))
        log1mp = F.logsigmoid(-margin)  # log(1 - sigmoid(m))

        loss_pos = -alpha * (1 - p) ** gamma * y * logp
        loss_neg = -(1 - alpha) * (p) ** gamma * (1 - y) * log1mp
        return (loss_pos + loss_neg).mean()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        alpha = 0.92
        gamma = 0.0
        YES_TOKEN_ID = 5081
        NO_TOKEN_ID = 708
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        mask = labels != -100
        label_pos = mask.float().argmax(dim=1)
        logits_for_loss = logits[mask]
        logit_yes = logits_for_loss[:, YES_TOKEN_ID]  # [B]
        logit_no = logits_for_loss[:, NO_TOKEN_ID]  # [B]

        margin = logit_yes - logit_no  # [B]
        reg = 1 - (logit_yes + logit_no)
        # Targets: infer y from the (single) label token
        B = labels.size(0)
        batch_idx = torch.arange(B, device=labels.device)
        target_tok = labels[batch_idx, label_pos]  # [B]
        y = (target_tok == YES_TOKEN_ID).long()
        loss = self.binary_focal_from_logits(margin, y, alpha=alpha, gamma=gamma) + reg
        # Define weights: [Weight for Exclude (0), Weight for Include (1)]
        # We give the positive class a weight of ~11.5
        # weights = torch.tensor([1.0, 11.5]).to(model.device)
        # weights = torch.ones((logits.shape[-1],), device=model.device)
        # weights[YES_TOKEN_ID] = 11.5 # yes token
        # weights[NO_TOKEN_ID] = 1 # no token
        # # Flatten logits and labels for CrossEntropy
        # loss_fct = nn.CrossEntropyLoss(weight=weights)
        # B, T, V = logits.shape
        # loss = loss_fct(logits.view(B*T,V), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class YesNoWeightedCETrainer(SFTTrainer):
    def __init__(
        self, *args, yes_id: int, no_id: int, pos_weight: float = 11.5, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.yes_id = int(yes_id)
        self.no_id = int(no_id)
        self.pos_weight = float(pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )
        logits = outputs.logits  # [B, T, V]

        # Find answer token position (assumes exactly one supervised token)
        comp_mask = labels != -100  # [B, T]
        comp_counts = comp_mask.sum(dim=1)
        if not torch.all(comp_counts == 1):
            raise ValueError(
                f"Expected 1 supervised token per example, got {comp_counts.tolist()}"
            )

        ans_pos = comp_mask.float().argmax(dim=1)  # label token position t
        pred_pos = ans_pos - 1  # logits position t-1
        if torch.any(pred_pos < 0):
            raise ValueError(
                "Answer token at position 0; cannot align next-token logits."
            )

        B = labels.size(0)
        idx = torch.arange(B, device=labels.device)
        step_logits = logits[idx, pred_pos, :]  # [B, V]

        # Extract 2-class logits
        logit_no = step_logits[:, self.no_id]
        logit_yes = step_logits[:, self.yes_id]
        logits_2 = torch.stack([logit_no, logit_yes], dim=1)  # [B, 2]

        # Target y: 1 if yes token, else 0
        target_tok = labels[idx, ans_pos]
        y = (target_tok == self.yes_id).long()

        # Weighted CE: weight positives
        # weights correspond to classes [no=0, yes=1]
        weights = torch.tensor([1.0, self.pos_weight], device=labels.device)
        loss = F.cross_entropy(logits_2, y, weight=weights)

        return (loss, outputs) if return_outputs else loss
