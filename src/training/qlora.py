import os
import random
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict
from functools import partial
import numpy as np

from datasets import Dataset, DatasetDict
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from tqdm import tqdm

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
    global y_true_list, y_pred_list, scores_list, explain_true_list, explain_pred_list
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
        ]  # Last token is the reason code digit (format: "yes/no , reason : R <code>")
        explain_true = labels[:, -1]  # Last token is the reason code digit
        n_token = 28711
        ra_token = 28741
        true_mask = (explain_true != n_token) & (explain_true != ra_token)
        explain_true = explain_true[
            true_mask
        ]  # Discard placeholder ('n') and RA tokens
        explain_pred = explain_pred[true_mask]
        explain_true_list.append(explain_true.numpy())
        explain_pred_list.append(explain_pred.numpy())

    if compute_result:
        y_true = np.concatenate(y_true_list, 0)
        y_pred = np.concatenate(y_pred_list, 0)
        scores = np.concatenate(scores_list, 0)
        explain_true = np.concatenate(explain_true_list, 0)
        explain_pred = np.concatenate(explain_pred_list)
        reason_macro_f1 = f1_score(
            explain_true, explain_pred, average="macro", zero_division=0
        )
        reason_cm = confusion_matrix(explain_true, explain_pred)
        print(f"Reason macro-F1: {reason_macro_f1:.4f}")
        print(f"Reason confusion matrix:\n{reason_cm}")
        metrics = {
            "accuracy": float((y_pred == y_true).mean()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "% to review for 95% recall": percent_to_review_for_recall(
                list(zip(y_pred, scores)), y_true, recall_target=0.95
            ),
            "average_precision": average_precision_score(y_true, scores),
            "explain_accuracy": float((explain_pred == explain_true).mean()),
            "reason_macro_f1": reason_macro_f1,
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
        positive_ratio: float = 0.3,
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
            positive_ratio=positive_ratio,
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
    def __init__(self, *args, positive_ratio: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_decision = 1.5
        self.lambda_reason = 1.0
        self.lambda_struct = 0.01
        self.positive_ratio = positive_ratio
        self._last_sub_losses: Dict[str, float] = {}
        self._eval_sub_loss_accum: Dict[str, float] = {}
        self._eval_sub_loss_count: int = 0

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        tokenizer = self.tokenizer

        self.model.eval()
        self.model.config.use_cache = True
        EvalObj = namedtuple("EvalObj", ["predictions", "label_ids"])

        all_metrics = None
        eval_loss_accum = {"loss_decision": 0.0, "loss_reason": 0.0, "loss_total": 0.0}
        eval_loss_count = 0

        for idx, example in enumerate(tqdm(eval_ds, desc="AR eval")):
            prompt = example["prompt"]
            completion = example["completion"]

            prompt_ids = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )["input_ids"]
            # Strip EOS if tokenizer appended it — we want the model to continue generating
            if prompt_ids[0, -1] == tokenizer.eos_token_id:
                prompt_ids = prompt_ids[:, :-1]
            prompt_ids = prompt_ids.to(self.model.device)
            attention_mask = torch.ones_like(prompt_ids)

            with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=6,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Stack generated-token scores into [1, gen_len, vocab]
            gen_logits = torch.stack(outputs.scores, dim=1)  # [1, gen_len, vocab]

            # Build label token ids for the completion
            label_ids = tokenizer(
                completion, add_special_tokens=False, return_tensors="pt"
            )[
                "input_ids"
            ]  # [1, comp_len]

            # Align lengths: generation may stop early (EOS) or label may be shorter
            gen_len = gen_logits.shape[1]
            label_len = label_ids.shape[1]
            if gen_len < label_len:
                # Pad logits with zeros (won't match any label, counted as wrong)
                pad = torch.zeros(
                    1,
                    label_len - gen_len,
                    gen_logits.shape[2],
                    device=gen_logits.device,
                )
                gen_logits = torch.cat([gen_logits, pad], dim=1)
            elif gen_len > label_len:
                gen_logits = gen_logits[:, :label_len, :]

            # Compute per-example AR eval losses
            label_ids_dev = label_ids.to(gen_logits.device)
            V = gen_logits.shape[-1]

            # Decision loss (first token = yes/no)
            weights_decision = torch.ones(V, device=gen_logits.device)
            weights_decision[5081] = 11.5  # yes
            weights_decision[708] = 1.0  # no
            loss_decision = F.cross_entropy(
                gen_logits[:, 0, :], label_ids_dev[:, 0], weight=weights_decision
            )

            # Reason loss (last token = reason code)
            RN_TOKEN_ID = 28711
            RA_TOKEN_ID = 28741
            reason_label = label_ids_dev[:, -1]
            if (reason_label != RN_TOKEN_ID).all() and (
                reason_label != RA_TOKEN_ID
            ).all():
                weights_reason = torch.ones(V, device=gen_logits.device)
                weights_reason[28734] = 6.4  # R0
                weights_reason[28740] = 3.9  # R1
                weights_reason[28750] = 9.6  # R2
                weights_reason[28770] = 3.1  # R3
                weights_reason[28781] = 1.0  # R4
                loss_reason = F.cross_entropy(
                    gen_logits[:, -1, :], reason_label, weight=weights_reason
                )
            else:
                loss_reason = torch.tensor(0.0, device=gen_logits.device)

            loss_total = (
                self.lambda_decision * loss_decision + self.lambda_reason * loss_reason
            )
            eval_loss_accum["loss_decision"] += loss_decision.item()
            eval_loss_accum["loss_reason"] += loss_reason.item()
            eval_loss_accum["loss_total"] += loss_total.item()
            eval_loss_count += 1

            eval_obj = EvalObj(predictions=gen_logits.cpu(), label_ids=label_ids)
            is_last = idx == len(eval_ds) - 1
            ret = compute_metrics(
                eval_obj,
                compute_result=is_last,
                explainability=True,
                tokenizer=tokenizer,
                shift=False,
            )
            if ret is not None:
                all_metrics = ret

        self.model.config.use_cache = False
        self.model.train()

        if all_metrics is None:
            all_metrics = {}

        if eval_loss_count > 0:
            all_metrics["ar_loss_decision"] = (
                eval_loss_accum["loss_decision"] / eval_loss_count
            )
            all_metrics["ar_loss_reason"] = (
                eval_loss_accum["loss_reason"] / eval_loss_count
            )
            all_metrics["ar_loss_total"] = (
                eval_loss_accum["loss_total"] / eval_loss_count
            )

        prefixed = {f"{metric_key_prefix}_{k}": v for k, v in all_metrics.items()}
        self.log(prefixed)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, prefixed
        )
        return prefixed

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        is_eval = any(k.startswith("eval_") for k in logs)
        if is_eval and self._eval_sub_loss_count > 0:
            avg = {
                f"eval_{k}": v / self._eval_sub_loss_count
                for k, v in self._eval_sub_loss_accum.items()
            }
            logs.update(avg)
            self._eval_sub_loss_accum = {}
            self._eval_sub_loss_count = 0
        elif not is_eval:
            logs.update(self._last_sub_losses)
        if start_time is not None:
            super().log(logs, start_time=start_time)
        else:
            super().log(logs)

    def get_train_dataloader(self):
        dataset = self.train_dataset
        # Determine per-example weight: positives get upweighted so they appear
        # at roughly `positive_ratio` frequency in each batch.
        is_positive = [
            "yes" in str(ex.get("completion", ex.get("labels", ""))) for ex in dataset
        ]
        n_pos = sum(is_positive)
        n_neg = len(is_positive) - n_pos
        if n_pos == 0 or n_neg == 0:
            return super().get_train_dataloader()
        r = self.positive_ratio
        w_pos = r / n_pos
        w_neg = (1.0 - r) / n_neg
        weights = [w_pos if p else w_neg for p in is_positive]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        logging.info(
            f"WeightedRandomSampler: {n_pos} positives, {n_neg} negatives, "
            f"target positive_ratio={r:.2f}"
        )
        data_collator = self.data_collator
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

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
        # Output format is now "yes/no, reason: <code>"
        # So: decision token is FIRST, reason code token is LAST

        # Inclusion decision: yes/no classification
        # Define weights: [Weight for Exclude (0), Weight for Include (1)]
        # We give the positive class a weight of ~11.5
        weights_decision = torch.ones((logits.shape[-1],), device=model.device)
        weights_decision[5081] = 11.5  # yes token
        weights_decision[708] = 1  # no token
        decision_logits = logits[:, 0]  # First generated token is the yes/no decision
        decision_labels = labels[:, 0]
        loss_include = F.cross_entropy(
            decision_logits.view(B, V),
            decision_labels.view(-1),
            weight=weights_decision,
        )

        # Reasoning decision: Select a decision for exclusion from the codebook
        # Only computed on excluded (no) examples — includes have placeholder Rn (28711)
        RN_TOKEN_ID = 28711  # placeholder token for included examples
        reason_logits = logits[
            :, -1
        ]  # Last token is the reason code digit (e.g. n, 0, 1...) — format is "yes/no , reason : R <code>"
        reason_labels = labels[:, -1]  # Last label token is the reason code digit

        RA_TOKEN_ID = 28741
        exclude_mask = (reason_labels != RN_TOKEN_ID) & (
            reason_labels != RA_TOKEN_ID
        )  # True for excluded examples with a specific reason
        weights_reason = torch.ones((logits.shape[-1],), device=model.device)
        # Inverse-frequency weights derived from validation class counts:
        # R0: 21, R1: 35, R2: 14, R3: 44, R4: 135, RA: 24
        # Scaled so R4 (majority) = 1.0
        weights_reason[28734] = 6.4  # R0 (135/21)
        weights_reason[28740] = 3.9  # R1 (135/35)
        weights_reason[28750] = 9.6  # R2 (135/14)
        weights_reason[28770] = 3.1  # R3 (135/44)
        weights_reason[28781] = 1.0  # R4 majority class
        # RA (28741) excluded from training — noisy label, handled via entropy at inference

        if exclude_mask.any():
            loss_reason = F.cross_entropy(
                reason_logits[exclude_mask].view(-1, V),
                reason_labels[exclude_mask].view(-1),
                weight=weights_reason,
            )
        else:
            loss_reason = torch.tensor(0.0, device=model.device)

        # Add a small overall loss to enforce structure (exclude decision and reason code)
        if T > 2:
            loss_struct = F.cross_entropy(
                logits[:, [1, 2, 3, 4], :].reshape(-1, V),
                labels[:, [1, 2, 3, 4]].reshape(-1),
            )
        else:
            loss_struct = torch.tensor(0.0, device=model.device)
        loss = (
            self.lambda_decision * loss_include
            + self.lambda_reason * loss_reason
            + self.lambda_struct * loss_struct
        )

        sub_losses = {
            "loss_decision": loss_include.detach().item(),
            "loss_reason": loss_reason.detach().item(),
            "loss_struct": loss_struct.detach().item(),
        }
        self._last_sub_losses = sub_losses
        if not self.model.training:
            for k, v in sub_losses.items():
                self._eval_sub_loss_accum[k] = self._eval_sub_loss_accum.get(k, 0.0) + v
            self._eval_sub_loss_count += 1

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
