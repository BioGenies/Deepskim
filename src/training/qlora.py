import os
import random
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, Union
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

from utils.evaluation import (
    REASON_TOKEN_IDS,
    REASON_ORDER,
    REASON_IDS_ORDERED,
    YES_ID,
    NO_ID,
    gather_reason_logprobs,
    compute_inclusion_prob,
    percent_to_review_for_recall,
)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

y_true_list = []
y_pred_list = []
scores_list = []
reason_true_list = []
reason_pred_list = []


def compute_metrics(eval_preds, compute_result, shift=True):
    """Compute decision and reason metrics from 3-token predictions (decision + reason).

    Each call accumulates one eval example. When compute_result=True,
    aggregates everything and returns the final metrics dict.
    """
    global y_true_list, y_pred_list, scores_list, reason_true_list, reason_pred_list

    logits = eval_preds.predictions  # [B, seq_len, vocab]  (torch, CPU)
    labels = eval_preds.label_ids  # [B, seq_len]

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    preds = logits.argmax(-1)

    if shift:
        preds = preds[:, :-1]
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]

    # First token: decision (yes/no), last token: reason digit
    pred_first = preds[:, 0]  # [B]
    label_first = labels[:, 0]  # [B]
    logits_first = logits[:, 0, :]  # [B, V]

    pred_last = preds[:, -1]  # [B]
    label_last = labels[:, -1]  # [B]

    # Binary decision from yes/no token
    y_true = (label_first == YES_ID).numpy().astype(int)
    y_pred = (pred_first == YES_ID).numpy().astype(int)

    # Ranking: P(yes) from 2-class softmax
    yes_no_logits = logits_first[:, [NO_ID, YES_ID]]  # [B, 2]
    p_yes = F.softmax(yes_no_logits.float(), dim=-1)[:, 1]  # [B]

    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    scores_list.append(p_yes.numpy())
    mask_rn = label_last != REASON_TOKEN_IDS["Rn"]
    label_last = label_last[mask_rn]
    pred_last = pred_last[mask_rn]
    if pred_last.numel() > 0:
        reason_true_list.append(label_last.numpy())
        reason_pred_list.append(pred_last.numpy())

    if compute_result:
        y_true_all = np.concatenate(y_true_list)
        y_pred_all = np.concatenate(y_pred_list)
        scores_all = np.concatenate(scores_list)
        reason_true_all = np.concatenate(reason_true_list)
        reason_pred_all = np.concatenate(reason_pred_list)

        reason_macro_f1 = f1_score(
            reason_true_all, reason_pred_all, average="macro", zero_division=0
        )
        reason_cm = confusion_matrix(
            reason_true_all,
            reason_pred_all,
            labels=[
                REASON_TOKEN_IDS["R0"],
                REASON_TOKEN_IDS["R1"],
                REASON_TOKEN_IDS["R2"],
                REASON_TOKEN_IDS["R3"],
                REASON_TOKEN_IDS["R4"],
                REASON_TOKEN_IDS["RA"],
                REASON_TOKEN_IDS["Rn"],
            ],
        )
        print(f"Reason macro-F1: {reason_macro_f1:.4f}")
        print(f"Reason confusion matrix:\n{reason_cm}")

        metrics = {
            "accuracy": float((y_pred_all == y_true_all).mean()),
            "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
            "recall": recall_score(y_true_all, y_pred_all, zero_division=0),
            "f1": f1_score(y_true_all, y_pred_all, zero_division=0),
            "% to review for 95% recall": percent_to_review_for_recall(
                list(zip(y_pred_all, scores_all)), y_true_all, recall_target=0.95
            ),
            "average_precision": average_precision_score(y_true_all, scores_all),
            "reason_accuracy": float((reason_pred_all == reason_true_all).mean()),
            "reason_macro_f1": reason_macro_f1,
        }

        y_true_list = []
        y_pred_list = []
        scores_list = []
        reason_true_list = []
        reason_pred_list = []
        return metrics


class QLora:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        lora_config: Dict[str, Any],
        sft_config: Dict[str, Any],
        train_dataset: Dataset,
        device: torch.device = torch.device("cpu"),
        eval_dataset: Dataset = None,
        positive_ratio: float = 0.3,
        label_smoothing: float = 0.03,
        continue_from: Union[str, None] = None,
    ):
        self.model = model
        self.device = device
        self.lora_config = LoraConfig(**lora_config)
        self.sft_config = SFTConfig(**sft_config)
        compute_metrics_func = partial(compute_metrics)
        self.trainer = ReasonCodeSFTTrainer(
            self.model,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics_func,
            args=self.sft_config,
            peft_config=self.lora_config,
            eval_dataset=eval_dataset,
            positive_ratio=positive_ratio,
            label_smoothing=label_smoothing,
        )
        self.continue_from = continue_from

    def train_model(self):
        assert self.trainer.train_dataset is not None, "Train dataset not set."
        if self.continue_from is not None:
            print(f"Resuming training from checkpoint: {self.continue_from}")
        self.trainer.train(self.continue_from)


class ReasonCodeSFTTrainer(SFTTrainer):
    """SFT trainer for decision + reason code classification.

    The model generates 3 tokens: decision (yes/no) + reason code (R0-R4, RA, Rn).
    Format: "no R1", "yes Rn", etc.
    """

    # sqrt of inverse-frequency ratios (R4 majority = 1.0) — gentler than
    # raw inverse-frequency, which was over-predicting the rarest class (R2)
    EXCL_REASON_ORDER = ["R0", "R1", "R2", "R3", "R4", "RA"]
    EXCL_REASON_IDS = [REASON_TOKEN_IDS[k] for k in EXCL_REASON_ORDER]
    EXCL_WEIGHTS = [2.53, 1.97, 3.10, 1.76, 1.0, 1.0]  # R0, R1, R2, R3, R4, RA
    POSITIVE_WEIGHT = 5
    LAMBDA_LOSSES = [
        0.2,
        1.0,
        0.01,
    ]  # Loss weights: decision, reason, structural

    def __init__(
        self,
        *args,
        positive_ratio=0.3,
        label_smoothing=0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.positive_ratio = positive_ratio
        self.label_smoothing = label_smoothing
        self._sub_loss_accum = {}
        self._sub_loss_count = 0
        self._eval_sub_loss_accum = {}
        self._eval_sub_loss_count = 0
        # Training metrics accumulators
        self._train_y_true = []
        self._train_y_pred = []
        self._train_scores = []
        self._train_reason_true = []
        self._train_reason_pred = []

    def get_train_dataloader(self):
        dataset = self.train_dataset
        is_positive = [
            str(ex.get("completion", "")).startswith("yes") for ex in dataset
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
            f"WeightedRandomSampler: {n_pos} positives (Rn), {n_neg} negatives, "
            f"target positive_ratio={r:.2f}"
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
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

        # Standard causal LM shift
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)

        # Select supervised tokens only
        mask = labels_flat != -100
        logits_select = logits_flat[mask]
        labels_select = labels_flat[mask]

        logits_2d = logits_select.view(B, -1, V)
        labels_2d = labels_select.view(B, -1)

        # Token positions: first = decision (yes/no), last = reason digit
        decision_logits = logits_2d[:, 0, :]  # [B, V]
        decision_labels = labels_2d[:, 0]  # [B]
        reason_logits = logits_2d[:, -1, :]  # [B, V]
        reason_labels = labels_2d[:, -1]  # [B]

        # ---- Part 1: Decision loss (yes/no, weighted) ----
        weights_decision = torch.ones(V, device=model.device)
        weights_decision[YES_ID] = self.POSITIVE_WEIGHT
        loss_decision = F.cross_entropy(
            decision_logits,
            decision_labels,
            weight=weights_decision,
            label_smoothing=self.label_smoothing,
        )

        # ---- Part 2: Conditional 6-class reason loss (excluded examples only) ----
        excl_mask = decision_labels == NO_ID
        loss_reason = torch.tensor(0.0, device=model.device)
        if excl_mask.any():
            excl_ids = torch.tensor(self.EXCL_REASON_IDS, device=model.device)
            excl_cls_logits = reason_logits[excl_mask][:, excl_ids]  # [n_excl, 6]
            excl_labels = reason_labels[excl_mask]  # [n_excl]
            # Map vocab token IDs → 0..5 index
            excl_label_idx = torch.zeros_like(excl_labels)
            for i, tid in enumerate(self.EXCL_REASON_IDS):
                excl_label_idx[excl_labels == tid] = i
            excl_weights = torch.tensor(
                self.EXCL_WEIGHTS, dtype=torch.float, device=model.device
            )
            loss_reason = F.cross_entropy(
                excl_cls_logits,
                excl_label_idx,
                weight=excl_weights,
                label_smoothing=self.label_smoothing,
            )

        # ---- Part 3: Structural loss on middle tokens (the " R" prefix) ----
        if logits_2d.shape[1] > 2:
            struct_logits = logits_2d[:, 1:-1, :].reshape(-1, V)
            struct_labels = labels_2d[:, 1:-1].reshape(-1)
            loss_struct = F.cross_entropy(struct_logits, struct_labels)
        else:
            loss_struct = torch.tensor(0.0, device=model.device)

        loss = (
            self.LAMBDA_LOSSES[0] * loss_decision
            + self.LAMBDA_LOSSES[1] * loss_reason
            + self.LAMBDA_LOSSES[2] * loss_struct
        )

        sub = {
            "loss_decision": loss_decision.detach().item(),
            "loss_reason": loss_reason.detach().item(),
            "loss_struct": loss_struct.detach().item(),
        }
        for k, v in sub.items():
            self._sub_loss_accum[k] = self._sub_loss_accum.get(k, 0.0) + v
        self._sub_loss_count += 1

        # Accumulate training metrics from teacher-forced logits
        with torch.no_grad():
            y_true = (decision_labels == YES_ID).cpu().numpy().astype(int)
            yes_no_logits = decision_logits[:, [NO_ID, YES_ID]]  # [B, 2]
            y_pred = yes_no_logits.argmax(-1).cpu().numpy().astype(int)
            p_yes = F.softmax(yes_no_logits.float(), dim=-1)[:, 1].cpu().numpy()
            self._train_y_true.append(y_true)
            self._train_y_pred.append(y_pred)
            self._train_scores.append(p_yes)
            self._train_reason_true.append(reason_labels.cpu().numpy())
            self._train_reason_pred.append(reason_logits.argmax(-1).cpu().numpy())

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        tokenizer = self.tokenizer

        self.model.eval()
        self.model.config.use_cache = True
        EvalObj = namedtuple("EvalObj", ["predictions", "label_ids"])

        all_metrics = None
        eval_loss_accum = {"decision": 0.0, "reason": 0.0}
        eval_loss_count = 0

        for idx, example in enumerate(tqdm(eval_ds, desc="AR eval")):
            prompt = example["prompt"]
            completion = example["completion"]

            prompt_ids = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )["input_ids"]
            if prompt_ids[0, -1] == tokenizer.eos_token_id:
                prompt_ids = prompt_ids[:, :-1]
            prompt_ids = prompt_ids.to(self.model.device)
            attention_mask = torch.ones_like(prompt_ids)

            with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=3,
                    min_new_tokens=3,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                input_length = prompt_ids.shape[1]
                generated_ids = outputs[0][..., input_length:]
                generated_ids = generated_ids.squeeze()
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                print(
                    f"Generated val response: {generated_text}, GT response: {completion}"
                )
            gen_logits = torch.stack(outputs.scores, dim=1)  # [1, gen_len, vocab]

            label_ids = tokenizer(
                completion, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]

            # Align lengths
            gen_len = gen_logits.shape[1]
            label_len = label_ids.shape[1]
            if gen_len < label_len:
                pad = torch.zeros(
                    1,
                    label_len - gen_len,
                    gen_logits.shape[2],
                    device=gen_logits.device,
                )
                gen_logits = torch.cat([gen_logits, pad], dim=1)
            elif gen_len > label_len:
                gen_logits = gen_logits[:, :label_len, :]

            # Decomposed eval loss matching training: decision + conditional reason
            label_ids_dev = label_ids.to(gen_logits.device)
            V = gen_logits.shape[-1]

            # Decision loss (first token = yes/no)
            first_logits = gen_logits[:, 0, :]  # [1, V]
            first_label = label_ids_dev[:, 0]  # [1]
            weights_decision = torch.ones(V, device=gen_logits.device)
            weights_decision[YES_ID] = self.POSITIVE_WEIGHT
            eval_loss_decision = F.cross_entropy(
                first_logits, first_label, weight=weights_decision
            )

            # Conditional reason loss (excluded examples only)
            eval_loss_reason = torch.tensor(0.0, device=gen_logits.device)
            if (first_label != YES_ID).all():
                last_logits = gen_logits[:, -1, :]  # [1, V]
                last_label = label_ids_dev[:, -1]  # [1]
                excl_ids = torch.tensor(self.EXCL_REASON_IDS, device=gen_logits.device)
                excl_cls_logits = last_logits[:, excl_ids]
                excl_label_idx = torch.zeros_like(last_label)
                for i, tid in enumerate(self.EXCL_REASON_IDS):
                    excl_label_idx[last_label == tid] = i
                excl_weights = torch.tensor(
                    self.EXCL_WEIGHTS, dtype=torch.float, device=gen_logits.device
                )
                eval_loss_reason = F.cross_entropy(
                    excl_cls_logits, excl_label_idx, weight=excl_weights
                )

            eval_loss_accum["decision"] += eval_loss_decision.item()
            eval_loss_accum["reason"] += eval_loss_reason.item()
            eval_loss_count += 1

            eval_obj = EvalObj(predictions=gen_logits.cpu(), label_ids=label_ids)
            is_last = idx == len(eval_ds) - 1
            ret = compute_metrics(eval_obj, compute_result=is_last, shift=False)
            if ret is not None:
                all_metrics = ret

        self.model.config.use_cache = False
        self.model.train()

        if all_metrics is None:
            all_metrics = {}
        if eval_loss_count > 0:
            all_metrics["ar_loss_decision"] = (
                eval_loss_accum["decision"] / eval_loss_count
            )
            all_metrics["ar_loss_reason"] = eval_loss_accum["reason"] / eval_loss_count
            all_metrics["ar_loss_total"] = (
                all_metrics["ar_loss_decision"] + all_metrics["ar_loss_reason"]
            )

        prefixed = {f"{metric_key_prefix}_{k}": v for k, v in all_metrics.items()}
        self.log(prefixed)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, prefixed
        )
        return prefixed

    def log(self, logs, start_time=None):
        is_eval = any(k.startswith("eval_") for k in logs)
        if not is_eval and self._sub_loss_count > 0:
            avg = {k: v / self._sub_loss_count for k, v in self._sub_loss_accum.items()}
            logs.update(avg)
            self._sub_loss_accum = {}
            self._sub_loss_count = 0

            # Compute and report training metrics
            if self._train_y_true:
                y_true_all = np.concatenate(self._train_y_true)
                y_pred_all = np.concatenate(self._train_y_pred)
                scores_all = np.concatenate(self._train_scores)
                reason_true_all = np.concatenate(self._train_reason_true)
                reason_pred_all = np.concatenate(self._train_reason_pred)

                logs["train_accuracy"] = float((y_pred_all == y_true_all).mean())
                logs["train_precision"] = precision_score(
                    y_true_all, y_pred_all, zero_division=0
                )
                logs["train_recall"] = recall_score(
                    y_true_all, y_pred_all, zero_division=0
                )
                logs["train_f1"] = f1_score(y_true_all, y_pred_all, zero_division=0)
                logs["train_average_precision"] = average_precision_score(
                    y_true_all, scores_all
                )
                logs["train_pct_review_95recall"] = percent_to_review_for_recall(
                    list(zip(y_pred_all, scores_all)), y_true_all, recall_target=0.95
                )
                logs["train_reason_accuracy"] = float(
                    (reason_pred_all == reason_true_all).mean()
                )
                logs["train_reason_macro_f1"] = f1_score(
                    reason_true_all, reason_pred_all, average="macro", zero_division=0
                )

                self._train_y_true = []
                self._train_y_pred = []
                self._train_scores = []
                self._train_reason_true = []
                self._train_reason_pred = []

        if start_time is not None:
            super().log(logs, start_time=start_time)
        else:
            super().log(logs)


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
    def __init__(
        self,
        *args,
        positive_ratio: float = 0.3,
        training_stage: int = 0,
        lambda_decision_anchor: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # training_stage: 0 = joint (default, legacy behaviour)
        #                 1 = decision-only (reason loss zeroed out)
        #                 2 = reason-only  (decision loss zeroed out, LoRA merged from stage 1)
        self.training_stage = training_stage
        if training_stage == 1:
            self.lambda_decision = 1.5
            self.lambda_reason = 0.0
            self.lambda_struct = 0.01
        elif training_stage == 2:
            self.lambda_decision = 0.0
            self.lambda_reason = 1.0
            self.lambda_struct = 0.0
            self.lambda_decision_anchor = lambda_decision_anchor
        else:
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
        labels_flat_orig = labels_flat  # keep for stage-2 anchor

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

        # Stage 2: decision anchor — KL-distill from the merged base model (LoRA disabled)
        # to prevent the new LoRA from drifting the decision boundary
        loss_anchor = torch.tensor(0.0, device=model.device)
        if self.training_stage == 2:
            YES_ID = 5081
            NO_ID = 708
            from peft import PeftModel as _PeftModel

            if isinstance(model, _PeftModel):
                # Get base model decision logits with LoRA disabled
                model.disable_adapter_layers()
                with torch.no_grad():
                    base_outputs = model(**inputs)
                model.enable_adapter_layers()
                base_logits = base_outputs.logits[..., :-1, :].contiguous()
                # Extract decision position from base logits
                B_orig = base_logits.shape[0]
                base_flat = base_logits.view(-1, base_logits.shape[-1])
                base_select = base_flat[labels_flat_orig != -100]
                base_decision = base_select.view(B_orig, -1, base_logits.shape[-1])[
                    :, 0, :
                ]
                # 2-class logits: [no, yes]
                base_2 = base_decision[:, [NO_ID, YES_ID]]
                curr_2 = decision_logits[:, [NO_ID, YES_ID]]
                # KL(base || current) — keep current close to base
                loss_anchor = F.kl_div(
                    F.log_softmax(curr_2, dim=-1),
                    F.softmax(base_2, dim=-1),
                    reduction="batchmean",
                )

        loss = (
            self.lambda_decision * loss_include
            + self.lambda_reason * loss_reason
            + self.lambda_struct * loss_struct
            + getattr(self, "lambda_decision_anchor", 0.0) * loss_anchor
        )

        sub_losses = {
            "loss_decision": loss_include.detach().item(),
            "loss_reason": loss_reason.detach().item(),
            "loss_struct": loss_struct.detach().item(),
            "loss_anchor": loss_anchor.detach().item(),
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
