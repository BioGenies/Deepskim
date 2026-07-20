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
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from utils.evaluation import (
    REASON_TOKEN_IDS,
    REASON_ORDER,
    REASON_IDS_ORDERED,
    YES_ID,
    NO_ID,
    MAYBE_ID,
    gather_reason_logprobs,
    compute_inclusion_prob,
    percent_to_review_for_recall,
    apply_reason_logit_bias,
    ReasonLogitBiasProcessor,
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
reason_gold_all_list = []  # gold reason per example (incl. Rn for positives)
maybe_score_list = []  # P(maybe) from the 3-way decision softmax (all rows)
maybe_gold_list = []  # gold == maybe (uncertainty target, all rows)

# R2 retired (Jul13): it was the absence-residual, not a content reason — 0 training rows.
EXCL_REASON_KEYS = ["R0", "R1", "R3", "R4"]


def _per_reason_classification_metrics(reason_true, reason_pred):
    """Per-class precision/recall/F1/support over R0..R4."""
    out = {}
    label_ids = [REASON_TOKEN_IDS[k] for k in EXCL_REASON_KEYS]
    p, r, f, s = precision_recall_fscore_support(
        reason_true,
        reason_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    for k, pk, rk, fk, sk in zip(EXCL_REASON_KEYS, p, r, f, s):
        out[f"reason_precision_{k}"] = float(pk)
        out[f"reason_recall_{k}"] = float(rk)
        out[f"reason_f1_{k}"] = float(fk)
        out[f"reason_support_{k}"] = int(sk)
    return out


def _stratified_ap_by_reason(y_true, scores, reason_gold):
    """AP per gold reason class on {positives} ∪ {negatives with gold k}.

    Diagnoses how well the decision head ranks positives above each
    sub-population of negatives. Skips classes with no negatives present.
    """
    out = {}
    pos_mask = y_true == 1
    for k in EXCL_REASON_KEYS:
        k_id = REASON_TOKEN_IDS[k]
        mask = pos_mask | (reason_gold == k_id)
        y = y_true[mask]
        s = scores[mask]
        if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
            continue
        ap = float(average_precision_score(y, s))
        # No-skill baseline = positive prevalence in the conditional subset.
        # Lift makes per-class APs comparable across strata with very different
        # negative counts (and to overall AP, which has its own prevalence).
        baseline = float(y.mean())
        out[f"average_precision_{k}"] = ap
        out[f"average_precision_lift_{k}"] = ap - baseline
    return out


def compute_metrics(eval_preds, compute_result, shift=True):
    """Compute decision and reason metrics from 3-token predictions (decision + reason).

    Each call accumulates one eval example. When compute_result=True,
    aggregates everything and returns the final metrics dict.
    """
    global y_true_list, y_pred_list, scores_list, reason_true_list, reason_pred_list, reason_gold_all_list, maybe_score_list, maybe_gold_list

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

    # First token: decision (yes/no/maybe), last token: reason digit
    pred_first = preds[:, 0]  # [B]
    label_first = labels[:, 0]  # [B]
    logits_first = logits[:, 0, :]  # [B, V]

    # Bias-corrected argmax for the reason token — undoes the residual
    # weighted-CE distortion that otherwise over-predicts high-weight classes
    pred_last = apply_reason_logit_bias(logits[:, -1, :]).argmax(-1)  # [B]
    label_last = labels[:, -1]  # [B]

    # --- Uncertainty: P(maybe) from the 3-way decision softmax vs gold==maybe,
    # over ALL rows (this is the whole point of the third decision token) ---
    three = logits_first[:, [YES_ID, NO_ID, MAYBE_ID]]  # [B, 3]
    p_maybe = F.softmax(three.float(), dim=-1)[:, 2]  # [B]
    maybe_score_list.append(p_maybe.numpy())
    maybe_gold_list.append((label_first == MAYBE_ID).numpy().astype(int))

    # --- Decision (include/exclude) + reason metrics on DECIDED rows only, so the
    # yes/no numbers stay directly comparable to the pre-maybe runs ---
    decided = (label_first == YES_ID) | (label_first == NO_ID)
    if decided.any():
        lf, pf = label_first[decided], pred_first[decided]
        y_true = (lf == YES_ID).numpy().astype(int)
        y_pred = (pf == YES_ID).numpy().astype(int)
        yes_no_logits = logits_first[decided][:, [NO_ID, YES_ID]]  # [n, 2]
        p_yes = F.softmax(yes_no_logits.float(), dim=-1)[:, 1]  # [n]
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        scores_list.append(p_yes.numpy())
        ll, pl = label_last[decided], pred_last[decided]
        reason_gold_all_list.append(ll.numpy())  # full vector, before Rn mask
        mask_rn = ll != REASON_TOKEN_IDS["Rn"]
        label_last_neg = ll[mask_rn]
        pred_last_neg = pl[mask_rn]
        if pred_last_neg.numel() > 0:
            reason_true_list.append(label_last_neg.numpy())
            reason_pred_list.append(pred_last_neg.numpy())

    if compute_result:
        y_true_all = np.concatenate(y_true_list)
        y_pred_all = np.concatenate(y_pred_list)
        scores_all = np.concatenate(scores_list)
        reason_true_all = np.concatenate(reason_true_list)
        reason_pred_all = np.concatenate(reason_pred_list)
        reason_gold_all = np.concatenate(reason_gold_all_list)

        # `labels=` is REQUIRED. Without it sklearn infers the class set from the union
        # of y_true and y_pred, so a single stray prediction of a class with no gold rows
        # (retired R2, or pred-only Rn on an excluded row) injects an F1=0 phantom class
        # and silently deflates the macro average by 1/k. Score the four REAL exclude
        # classes only. See [[reason-macro-f1-divide-by-six-artifact]].
        excl_label_ids = [REASON_TOKEN_IDS[k] for k in EXCL_REASON_KEYS]
        reason_macro_f1 = f1_score(
            reason_true_all,
            reason_pred_all,
            labels=excl_label_ids,
            average="macro",
            zero_division=0,
        )
        # Rn kept in the confusion matrix only to SEE pred-only Rn leakage on excludes.
        reason_cm = confusion_matrix(
            reason_true_all,
            reason_pred_all,
            labels=excl_label_ids + [REASON_TOKEN_IDS["Rn"]],
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
        metrics.update(
            _stratified_ap_by_reason(y_true_all, scores_all, reason_gold_all)
        )
        metrics.update(
            _per_reason_classification_metrics(reason_true_all, reason_pred_all)
        )

        # Uncertainty head-to-head number: how well P(maybe) ranks the gold
        # 'maybe' (unclear) rows above the confident yes/no rows.
        maybe_score_all = np.concatenate(maybe_score_list)
        maybe_gold_all = np.concatenate(maybe_gold_list)
        metrics["uncertainty_prevalence"] = float(maybe_gold_all.mean())
        if maybe_gold_all.sum() > 0:
            metrics["uncertainty_ap"] = float(
                average_precision_score(maybe_gold_all, maybe_score_all)
            )
        if 0 < maybe_gold_all.sum() < len(maybe_gold_all):
            metrics["uncertainty_auc"] = float(
                roc_auc_score(maybe_gold_all, maybe_score_all)
            )

        y_true_list = []
        y_pred_list = []
        scores_list = []
        reason_true_list = []
        reason_pred_list = []
        reason_gold_all_list = []
        maybe_score_list = []
        maybe_gold_list = []
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
        reason_weights: Union[Dict[str, float], None] = None,
        maybe_weight: Union[float, None] = None,
        maybe_ratio: Union[float, None] = None,
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
            reason_weights=reason_weights,
            maybe_weight=maybe_weight,
            maybe_ratio=maybe_ratio,
        )
        self.continue_from = continue_from

    def train_model(self):
        assert self.trainer.train_dataset is not None, "Train dataset not set."
        if self.continue_from is not None:
            print(f"Resuming training from checkpoint: {self.continue_from}")
        self.trainer.train(self.continue_from)


class ReasonCodeSFTTrainer(SFTTrainer):
    """SFT trainer for decision + reason code classification.

    The model generates 3 tokens: decision (yes/no) + reason code (R0-R4, Rn).
    Format: "no R1", "yes Rn", etc.
    """

    # sqrt(majority/n_c) over the CURRENT (v5) train exclude counts:
    #   R0=546, R1=297, R3=170, R4=671 (majority).  R2 retired (0 rows).
    # WAS [2.63, 1.89, 1.54, 2.09, 1.0] from a long-dead corpus (R0=73 ... R4=506) in
    # which R0 was the RAREST class; in v5 R0 is the second most COMMON, so that vector
    # up-weighted a common class 2.6x in the reason CE. Recompute when splits change,
    # and keep in sync with REASON_TRAIN_WEIGHTS in evaluation.py (the inference de-bias).
    EXCL_REASON_ORDER = ["R0", "R1", "R3", "R4"]
    EXCL_REASON_IDS = [REASON_TOKEN_IDS[k] for k in EXCL_REASON_ORDER]
    EXCL_WEIGHTS = [1.11, 1.50, 1.99, 1.0]  # R0, R1, R3, R4
    POSITIVE_WEIGHT = 5
    MAYBE_WEIGHT = 5  # up-weight the rare 'maybe' decision token in the CE
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
        reason_weights: Union[Dict[str, float], None] = None,
        maybe_weight: Union[float, None] = None,
        maybe_ratio: Union[float, None] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.positive_ratio = positive_ratio
        self.label_smoothing = label_smoothing
        self.reason_weights = reason_weights
        # 'maybe' (uncertainty) decision token controls
        self.maybe_weight = (
            float(maybe_weight) if maybe_weight is not None else float(self.MAYBE_WEIGHT)
        )
        self.maybe_ratio = maybe_ratio  # target batch fraction; None -> natural rate
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
        self._train_reason_gold = []  # full gold reason per example, incl. Rn
        self._train_maybe_score = []  # P(maybe) on train micro-batches
        self._train_maybe_gold = []  # gold == maybe on train micro-batches
        # --- timing probe: instruments the first N micro-batches only ---
        self._probe_max_steps = 30
        self._probe_step = 0
        self._probe_fwd_ms = 0.0

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not torch.cuda.is_available() or self._probe_step >= self._probe_max_steps:
            return super().training_step(model, inputs, num_items_in_batch)

        ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        B, T = (int(ids.shape[0]), int(ids.shape[1])) if ids is not None else (-1, -1)
        n_sup = int((labels != -100).sum().item()) if labels is not None else -1

        self._probe_fwd_ms = 0.0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = super().training_step(model, inputs, num_items_in_batch)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        bwd_ms = total_ms - self._probe_fwd_ms
        self._probe_step += 1
        logging.info(
            f"[probe] microbatch {self._probe_step}/{self._probe_max_steps} "
            f"B={B} T={T} sup_tokens={n_sup} tokens={B * T} | "
            f"total={total_ms:.0f}ms fwd={self._probe_fwd_ms:.0f}ms bwd={bwd_ms:.0f}ms"
        )
        return loss

    def get_train_dataloader(self):
        dataset = self.train_dataset

        def _category(ex):
            """(class, reason) — class in {pos(include), maybe(unclear), neg(exclude)}."""
            comp = str(ex.get("completion", ""))
            if comp.startswith("yes"):
                return "pos", None
            if comp.startswith("maybe"):
                return "maybe", None
            parts = comp.split()
            return "neg", (parts[-1] if parts else None)

        cats = [_category(ex) for ex in dataset]
        N = len(cats)
        n_pos = sum(1 for c0, _ in cats if c0 == "pos")
        n_maybe = sum(1 for c0, _ in cats if c0 == "maybe")
        n_neg = N - n_pos - n_maybe
        if n_pos == 0 or n_neg == 0:
            return super().get_train_dataloader()

        r = self.positive_ratio
        # Target batch fraction for 'maybe'; default = natural rate (keep prevalence)
        # unless explicitly up-sampled via maybe_ratio. Leave ≥5% for negatives.
        r_maybe = (
            float(self.maybe_ratio)
            if self.maybe_ratio is not None
            else (n_maybe / N if n_maybe else 0.0)
        )
        r_maybe = min(r_maybe, max(0.0, 1.0 - r - 0.05))
        r_neg = max(0.0, 1.0 - r - r_maybe)
        w_pos = r / n_pos if n_pos else 0.0
        w_maybe = r_maybe / n_maybe if n_maybe else 0.0

        # Per-reason multipliers (default 1.0 → reproduces uniform per-example
        # negative weighting, i.e. natural class frequency within negatives).
        m = {k: 1.0 for k in self.EXCL_REASON_ORDER}
        if self.reason_weights:
            for k, v in self.reason_weights.items():
                m[k] = float(v)

        n_by_reason: Dict[str, int] = {}
        for c0, rk in cats:
            if c0 != "neg" or rk is None:
                continue
            n_by_reason[rk] = n_by_reason.get(rk, 0) + 1

        # Z normalises so that Σ over negatives of (m_k * r_neg / Z) = r_neg.
        Z = sum(m.get(k, 1.0) * n for k, n in n_by_reason.items())
        if Z <= 0:
            return super().get_train_dataloader()

        weights = []
        for c0, rk in cats:
            if c0 == "pos":
                weights.append(w_pos)
            elif c0 == "maybe":
                weights.append(w_maybe)
            else:
                weights.append(m.get(rk, 1.0) * r_neg / Z)

        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        # Realised conditional share P(reason=k | negative) in a sampled batch.
        realised = {k: m.get(k, 1.0) * n / Z for k, n in n_by_reason.items()}
        logging.info(
            f"WeightedRandomSampler: {n_pos} positives (Rn), {n_maybe} maybe, "
            f"{n_neg} negatives, target positive_ratio={r:.2f}, maybe_ratio={r_maybe:.2f}, "
            f"reason counts={n_by_reason}, "
            f"reason multipliers={m}, "
            f"realised P(reason|neg)={ {k: round(v, 3) for k, v in realised.items()} }"
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
        if torch.cuda.is_available() and self._probe_step < self._probe_max_steps:
            _s = torch.cuda.Event(enable_timing=True)
            _e = torch.cuda.Event(enable_timing=True)
            _s.record()
            outputs = model(**inputs)
            _e.record()
            torch.cuda.synchronize()
            self._probe_fwd_ms = _s.elapsed_time(_e)
        else:
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

        # ---- Part 1: Decision loss (yes/no/maybe, weighted) ----
        # Full-vocab CE, so the third gold class ('maybe') is handled directly;
        # only its class weight is added.
        weights_decision = torch.ones(V, device=model.device)
        weights_decision[YES_ID] = self.POSITIVE_WEIGHT
        weights_decision[MAYBE_ID] = self.maybe_weight
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
            # Uncertainty: P(maybe) vs gold==maybe over all rows in the micro-batch
            three = decision_logits[:, [YES_ID, NO_ID, MAYBE_ID]]  # [B, 3]
            self._train_maybe_score.append(
                F.softmax(three.float(), dim=-1)[:, 2].cpu().numpy()
            )
            self._train_maybe_gold.append(
                (decision_labels == MAYBE_ID).cpu().numpy().astype(int)
            )
            # Decision + reason metrics on decided (yes/no) rows only, so the
            # include/exclude numbers stay comparable to the pre-maybe runs.
            decided = (decision_labels == YES_ID) | (decision_labels == NO_ID)
            if decided.any():
                dlab = decision_labels[decided]
                yes_no_logits = decision_logits[decided][:, [NO_ID, YES_ID]]  # [n, 2]
                self._train_y_true.append((dlab == YES_ID).cpu().numpy().astype(int))
                self._train_y_pred.append(
                    yes_no_logits.argmax(-1).cpu().numpy().astype(int)
                )
                self._train_scores.append(
                    F.softmax(yes_no_logits.float(), dim=-1)[:, 1].cpu().numpy()
                )
                self._train_reason_gold.append(reason_labels[decided].cpu().numpy())
            # Mirror eval: score reason only on excluded examples (positives
            # are untrained on Rn) and apply bias correction before argmax.
            excl_mask_metric = decision_labels == NO_ID
            if excl_mask_metric.any():
                corrected = apply_reason_logit_bias(reason_logits[excl_mask_metric])
                self._train_reason_true.append(
                    reason_labels[excl_mask_metric].cpu().numpy()
                )
                self._train_reason_pred.append(corrected.argmax(-1).cpu().numpy())

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        tokenizer = self.tokenizer

        self.model.eval()
        self.model.config.use_cache = True
        EvalObj = namedtuple("EvalObj", ["predictions", "label_ids"])

        # Infer the " R" prefix token id (the token that immediately precedes
        # the reason digit in the completion format) so the bias processor
        # fires on exactly the reason step.
        r_prefix_id = int(tokenizer(" R", add_special_tokens=False)["input_ids"][-1])
        bias_processor = ReasonLogitBiasProcessor(r_prefix_token_id=r_prefix_id)

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
                    logits_processor=[bias_processor],
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

            # Decision loss (first token = yes/no/maybe)
            first_logits = gen_logits[:, 0, :]  # [1, V]
            first_label = label_ids_dev[:, 0]  # [1]
            weights_decision = torch.ones(V, device=gen_logits.device)
            weights_decision[YES_ID] = self.POSITIVE_WEIGHT
            weights_decision[MAYBE_ID] = self.maybe_weight
            eval_loss_decision = F.cross_entropy(
                first_logits, first_label, weight=weights_decision
            )

            # Conditional reason loss (excluded 'no' examples only — not yes/maybe)
            eval_loss_reason = torch.tensor(0.0, device=gen_logits.device)
            if (first_label == NO_ID).all():
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
                if self._train_reason_gold:
                    reason_gold_all = np.concatenate(self._train_reason_gold)
                    strat = _stratified_ap_by_reason(
                        y_true_all, scores_all, reason_gold_all
                    )
                    for k, v in strat.items():
                        logs[f"train_{k}"] = v
                if self._train_reason_true:
                    reason_true_all = np.concatenate(self._train_reason_true)
                    reason_pred_all = np.concatenate(self._train_reason_pred)
                    logs["train_reason_accuracy"] = float(
                        (reason_pred_all == reason_true_all).mean()
                    )
                    logs["train_reason_macro_f1"] = f1_score(
                        reason_true_all,
                        reason_pred_all,
                        average="macro",
                        zero_division=0,
                    )
                    per_class = _per_reason_classification_metrics(
                        reason_true_all, reason_pred_all
                    )
                    for k, v in per_class.items():
                        logs[f"train_{k}"] = v

                self._train_y_true = []
                self._train_y_pred = []
                self._train_scores = []
                self._train_reason_true = []
                self._train_reason_pred = []
                self._train_reason_gold = []

            # Train-side uncertainty (P(maybe)) — computed independently of the
            # decided-row metrics so it logs every window; a window with no
            # 'maybe' rows skips AP/AUC cleanly instead of logging a misleading
            # 0.0 (at the real batch size both always populate).
            if self._train_maybe_score:
                ms = np.concatenate(self._train_maybe_score)
                mg = np.concatenate(self._train_maybe_gold)
                if mg.sum() > 0:
                    logs["train_uncertainty_ap"] = float(
                        average_precision_score(mg, ms)
                    )
                if 0 < mg.sum() < len(mg):
                    logs["train_uncertainty_auc"] = float(roc_auc_score(mg, ms))
                self._train_maybe_score = []
                self._train_maybe_gold = []

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
