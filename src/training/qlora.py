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

EXCL_REASON_KEYS = ["R0", "R1", "R2", "R3", "R4"]


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
    global y_true_list, y_pred_list, scores_list, reason_true_list, reason_pred_list, reason_gold_all_list

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

    # Bias-corrected argmax for the reason token — undoes the residual
    # weighted-CE distortion that otherwise over-predicts high-weight classes
    pred_last = apply_reason_logit_bias(logits[:, -1, :]).argmax(-1)  # [B]
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
    reason_gold_all_list.append(label_last.numpy())  # full vector, before Rn mask
    mask_rn = label_last != REASON_TOKEN_IDS["Rn"]
    label_last_neg = label_last[mask_rn]
    pred_last_neg = pred_last[mask_rn]
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
        metrics.update(
            _stratified_ap_by_reason(y_true_all, scores_all, reason_gold_all)
        )
        metrics.update(
            _per_reason_classification_metrics(reason_true_all, reason_pred_all)
        )

        y_true_list = []
        y_pred_list = []
        scores_list = []
        reason_true_list = []
        reason_pred_list = []
        reason_gold_all_list = []
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
        continue_from: Union[str, None] = None,
        lambda_suff: float = 0.0,
        suff_pos_weight: Union[float, None] = None,
        suff_dropout: float = 0.0,
        suff_pooling: str = "mean",
        suff_detach: bool = False,
        suff_hidden_dim: Union[int, None] = None,
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
            lambda_suff=lambda_suff,
            suff_pos_weight=suff_pos_weight,
            suff_dropout=suff_dropout,
            suff_pooling=suff_pooling,
            suff_detach=suff_detach,
            suff_hidden_dim=suff_hidden_dim,
        )
        self.continue_from = continue_from

    def train_model(self):
        assert self.trainer.train_dataset is not None, "Train dataset not set."
        if self.continue_from is not None:
            print(f"Resuming training from checkpoint: {self.continue_from}")
        self.trainer.train(self.continue_from)


def pool_prompt_hidden(h, prompt_mask, mode="mean"):
    """Pool per-token hidden states over the PROMPT tokens only.

    Args:
        h: [B, T, H] hidden states (un-shifted).
        prompt_mask: [B, T] bool, True on prompt (non-pad, non-completion) tokens.
        mode: "mean" (masked mean over prompt tokens, the legacy behaviour) or
              "last" (the final prompt token — the position whose next-token
              prediction IS the yes/no decision, so it carries the decision-shaped
              summary; padding-side agnostic via a max-index gather).
    Returns:
        [B, H] pooled features.
    """
    if mode == "last":
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0).expand_as(prompt_mask)
        masked = torch.where(prompt_mask, pos, torch.full_like(pos, -1))
        idx = masked.max(dim=1).values.clamp(min=0)  # [B]
        return h[torch.arange(h.shape[0], device=h.device), idx]
    pm = prompt_mask.unsqueeze(-1).to(h.dtype)
    return (h * pm).sum(1) / pm.sum(1).clamp(min=1.0)


class SufficiencyHead(nn.Module):
    """Probe on the pooled prompt hidden states -> logit P(insufficient).

    Input dropout regularises the (hidden_size)-dim pooled features. The head
    overfits fast at the ~4-5% `unclear` prevalence (only ~90 train positives) —
    the first no-dropout run peaked ~step 300 then collapsed — so dropout is the
    primary defence and a bare linear map is the default. `hidden_dim=None` gives
    the original `nn.Linear` with state_dict keys `linear.{weight,bias}` (what the
    serving/eval path reloads). Setting `hidden_dim` swaps in a small bottleneck
    MLP (keys `net.*`) — more capacity, higher overfit risk; use only as an
    ablation, not the default. Pooling is chosen upstream (see pool_prompt_hidden).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0, hidden_dim=None):
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        if hidden_dim:
            self.linear = None
            self.net = nn.Sequential(
                nn.Linear(hidden_size, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.linear = nn.Linear(hidden_size, 1)
            self.net = None

    def forward(self, x):
        x = self.dropout(x)
        return self.net(x) if self.net is not None else self.linear(x)


class ReasonCodeSFTTrainer(SFTTrainer):
    """SFT trainer for decision + reason code classification.

    The model generates 3 tokens: decision (yes/no) + reason code (R0-R4, Rn).
    Format: "no R1", "yes Rn", etc.
    """

    # sqrt of inverse-frequency over post-cleanup train counts
    # (R0=73, R1=141, R2=212, R3=116, R4=506), R4 majority = 1.0.
    EXCL_REASON_ORDER = ["R0", "R1", "R2", "R3", "R4"]
    EXCL_REASON_IDS = [REASON_TOKEN_IDS[k] for k in EXCL_REASON_ORDER]
    EXCL_WEIGHTS = [2.63, 1.89, 1.54, 2.09, 1.0]  # R0, R1, R2, R3, R4
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
        reason_weights: Union[Dict[str, float], None] = None,
        lambda_suff: float = 0.0,
        suff_pos_weight: Union[float, None] = None,
        suff_dropout: float = 0.0,
        suff_pooling: str = "mean",
        suff_detach: bool = False,
        suff_hidden_dim: Union[int, None] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.positive_ratio = positive_ratio
        self.label_smoothing = label_smoothing
        self.reason_weights = reason_weights
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
        self._train_suff_scores = []  # dropout-free head scores for train AUC/AP
        self._train_suff_labels = []
        # --- timing probe: instruments the first N micro-batches only ---
        self._probe_max_steps = 30
        self._probe_step = 0
        self._probe_fwd_ms = 0.0

        # --- sufficiency head (Option B): linear head on the mean-pooled prompt
        # hidden states, trained with weighted BCE on the per-example
        # `sufficiency` label. lambda_suff <= 0 disables the HEAD (legacy
        # decision+reason model), but `unclear` rows are still masked from the
        # decision/reason loss regardless — see the collator wrap below. ---
        self.lambda_suff = float(lambda_suff or 0.0)
        self.suff_pos_weight = suff_pos_weight
        self.suff_dropout = float(suff_dropout or 0.0)
        # `last` pooling reads the final prompt token (the decision-generation
        # state); `detach` stops the sufficiency loss from back-propagating into
        # the shared backbone, so it becomes a pure readout that cannot perturb
        # the decision (the decision objective already makes that token
        # sufficiency-rich). hidden_dim swaps the linear head for a bottleneck MLP.
        self.suff_pooling = str(suff_pooling or "mean")
        self.suff_detach = bool(suff_detach)
        self.suff_hidden_dim = suff_hidden_dim
        self.suff_head = None
        if self.lambda_suff > 0.0:
            hidden = self.model.config.hidden_size
            dev = next(self.model.parameters()).device
            head = SufficiencyHead(
                hidden, dropout=self.suff_dropout, hidden_dim=self.suff_hidden_dim
            ).to(device=dev, dtype=torch.float32)
            for p in head.parameters():
                p.requires_grad_(True)
            # register as a submodule of the (PEFT) model so it is both moved
            # with the model, toggled by model.train()/eval() (so dropout is
            # active in training and off at eval), and picked up by the Trainer
            # optimizer.
            self.model.suff_head = head
            self.suff_head = head
            logging.info(
                f"Sufficiency head attached (hidden={hidden}, lambda_suff="
                f"{self.lambda_suff}, dropout={self.suff_dropout}, "
                f"pos_weight={self.suff_pos_weight}, pooling={self.suff_pooling}, "
                f"detach={self.suff_detach}, hidden_dim={self.suff_hidden_dim})"
            )

        # Wrap the data collator UNCONDITIONALLY to surface the scalar
        # `sufficiency` label. It drives the decision/reason masking of `unclear`
        # rows in compute_loss, which must happen whether or not the head is
        # attached — otherwise the lambda=0 baseline would train on the unclear
        # placeholder completions and stop being a clean ablation. The base LM
        # collator would drop this extra scalar field.
        base_collator = self.data_collator

        def _suff_collator(features, _base=base_collator):
            suff = None
            if features and "sufficiency" in features[0]:
                suff = [int(f.pop("sufficiency")) for f in features]
            batch = _base(features)
            if suff is not None:
                batch["sufficiency"] = torch.tensor(suff, dtype=torch.float)
            return batch

        self.data_collator = _suff_collator

    def _save(self, output_dir=None, state_dict=None):
        # Persist the sufficiency head next to the PEFT adapter (PeftModel
        # save_pretrained does not capture the extra head).
        super()._save(output_dir, state_dict)
        if self.suff_head is not None:
            out = output_dir if output_dir is not None else self.args.output_dir
            torch.save(self.suff_head.state_dict(), os.path.join(out, "suff_head.pt"))

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

        def _reason_of(ex):
            comp = str(ex.get("completion", ""))
            if comp.startswith("yes"):
                return None
            parts = comp.split()
            return parts[-1] if parts else None

        reasons = [_reason_of(ex) for ex in dataset]
        is_positive = [r is None for r in reasons]
        n_pos = sum(is_positive)
        n_neg = len(is_positive) - n_pos
        if n_pos == 0 or n_neg == 0:
            return super().get_train_dataloader()
        r = self.positive_ratio
        w_pos = r / n_pos

        # Per-reason multipliers (default 1.0 → reproduces uniform per-example
        # negative weighting, i.e. natural class frequency within negatives).
        m = {k: 1.0 for k in self.EXCL_REASON_ORDER}
        if self.reason_weights:
            for k, v in self.reason_weights.items():
                m[k] = float(v)

        n_by_reason: Dict[str, int] = {}
        for rk in reasons:
            if rk is None:
                continue
            n_by_reason[rk] = n_by_reason.get(rk, 0) + 1

        # Z normalises so that Σ over negatives of (m_k * (1-r) / Z) = 1 - r.
        Z = sum(m.get(k, 1.0) * n for k, n in n_by_reason.items())
        if Z <= 0:
            return super().get_train_dataloader()

        weights = []
        for is_pos, rk in zip(is_positive, reasons):
            if is_pos:
                weights.append(w_pos)
            else:
                weights.append(m.get(rk, 1.0) * (1.0 - r) / Z)

        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        # Realised conditional share P(reason=k | negative) in a sampled batch.
        realised = {k: m.get(k, 1.0) * n / Z for k, n in n_by_reason.items()}
        logging.info(
            f"WeightedRandomSampler: {n_pos} positives (Rn), {n_neg} negatives, "
            f"target positive_ratio={r:.2f}, "
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
        # `sufficiency` is not a model input — pop it before the forward pass.
        suff_labels = inputs.pop("sufficiency", None)
        ohs = self.suff_head is not None  # need hidden states for the suff head
        if torch.cuda.is_available() and self._probe_step < self._probe_max_steps:
            _s = torch.cuda.Event(enable_timing=True)
            _e = torch.cuda.Event(enable_timing=True)
            _s.record()
            outputs = model(**inputs, output_hidden_states=ohs)
            _e.record()
            torch.cuda.synchronize()
            self._probe_fwd_ms = _s.elapsed_time(_e)
        else:
            outputs = model(**inputs, output_hidden_states=ohs)
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

        # `unclear` rows (sufficiency==1) carry only a placeholder leaning
        # completion — supervise the sufficiency head on them but mask their
        # decision/reason loss so a guessed direction never trains those heads.
        if suff_labels is not None:
            decided = suff_labels.to(decision_labels.device) == 0
        else:
            decided = torch.ones_like(decision_labels, dtype=torch.bool)

        # ---- Part 1: Decision loss (yes/no, weighted; decided rows only) ----
        weights_decision = torch.ones(V, device=model.device)
        weights_decision[YES_ID] = self.POSITIVE_WEIGHT
        if decided.any():
            loss_decision = F.cross_entropy(
                decision_logits[decided],
                decision_labels[decided],
                weight=weights_decision,
                label_smoothing=self.label_smoothing,
            )
        else:
            loss_decision = torch.tensor(0.0, device=model.device)

        # ---- Part 2: Conditional 6-class reason loss (excluded, decided only) ----
        excl_mask = (decision_labels == NO_ID) & decided
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

        # ---- Part 4: Sufficiency head — weighted BCE on the pooled PROMPT hidden
        # states (pooling over prompt tokens only avoids leaking the completion).
        # `last` pooling reads the decision-generation token; `detach` makes the
        # head a pure readout that never perturbs the shared backbone. ----
        loss_suff = torch.tensor(0.0, device=model.device)
        if suff_labels is not None and self.suff_head is not None:
            h = outputs.hidden_states[-1]  # [B, T, H], un-shifted
            am = inputs["attention_mask"]
            full_labels = inputs["labels"]
            prompt_mask = (full_labels == -100) & (am == 1)  # [B, T] bool
            pooled = pool_prompt_hidden(h, prompt_mask, mode=self.suff_pooling)
            if self.suff_detach:
                pooled = pooled.detach()
            suff_logit = self.suff_head(pooled.float()).squeeze(-1)  # [B]
            pw = (
                torch.tensor([self.suff_pos_weight], device=suff_logit.device)
                if self.suff_pos_weight
                else None
            )
            loss_suff = F.binary_cross_entropy_with_logits(
                suff_logit, suff_labels.float().to(suff_logit.device), pos_weight=pw
            )
            # Accumulate train sufficiency scores/labels for train AUC/AP (logged in
            # log()). Recompute the logit with the head in eval mode so dropout is
            # off — makes the train metric comparable to the dev metric (dropout-free)
            # and a clean read of the train fit vs the 0.88 linear-probe dev ceiling.
            with torch.no_grad():
                was_training = self.suff_head.training
                self.suff_head.eval()
                clean_logit = self.suff_head(pooled.detach().float()).squeeze(-1)
                if was_training:
                    self.suff_head.train()
                self._train_suff_scores.append(
                    torch.sigmoid(clean_logit).cpu().numpy()
                )
                self._train_suff_labels.append(
                    suff_labels.detach().cpu().numpy().astype(int)
                )

        loss = (
            self.LAMBDA_LOSSES[0] * loss_decision
            + self.LAMBDA_LOSSES[1] * loss_reason
            + self.LAMBDA_LOSSES[2] * loss_struct
            + self.lambda_suff * loss_suff
        )

        sub = {
            "loss_decision": loss_decision.detach().item(),
            "loss_reason": loss_reason.detach().item(),
            "loss_struct": loss_struct.detach().item(),
        }
        if self.suff_head is not None:
            sub["loss_suff"] = loss_suff.detach().item()
        for k, v in sub.items():
            self._sub_loss_accum[k] = self._sub_loss_accum.get(k, 0.0) + v
        self._sub_loss_count += 1

        # Accumulate training metrics from teacher-forced logits (decided rows
        # only — `unclear` placeholders would otherwise pollute decision metrics).
        with torch.no_grad():
            dec = decided.cpu().numpy().astype(bool)
            y_true = (decision_labels == YES_ID).cpu().numpy().astype(int)[dec]
            yes_no_logits = decision_logits[:, [NO_ID, YES_ID]]  # [B, 2]
            y_pred = yes_no_logits.argmax(-1).cpu().numpy().astype(int)[dec]
            p_yes = F.softmax(yes_no_logits.float(), dim=-1)[:, 1].cpu().numpy()[dec]
            self._train_y_true.append(y_true)
            self._train_y_pred.append(y_pred)
            self._train_scores.append(p_yes)
            self._train_reason_gold.append(reason_labels.cpu().numpy()[dec])
            # Mirror eval: score reason only on excluded examples (positives
            # are untrained on Rn) and apply bias correction before argmax.
            excl_mask_metric = (decision_labels == NO_ID) & decided
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
        suff_scores: list[float] = []
        suff_labels_eval: list[int] = []
        # `unclear` rows are undecidable — their placeholder completion must not
        # pollute the decision/reason metrics; they feed the sufficiency AUC only.
        # Aggregate compute_metrics on the last DECIDED row (not the last row).
        try:
            suff_flags = [int(s) for s in eval_ds["sufficiency"]]
        except (KeyError, TypeError):
            suff_flags = [0] * len(eval_ds)
        last_decided_idx = max(
            (i for i, s in enumerate(suff_flags) if s == 0), default=-1
        )

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

            # Sufficiency head: p_insufficient from the pooled prompt hidden states
            # (prompt-only input, so pooling matches training via pool_prompt_hidden:
            # mean over all prompt tokens, or the final prompt token for `last`).
            if self.suff_head is not None:
                with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
                    ho = self.model(
                        input_ids=prompt_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    pooled = pool_prompt_hidden(
                        ho.hidden_states[-1], attention_mask.bool(), mode=self.suff_pooling
                    )  # [1, H]
                    p_insuff = torch.sigmoid(
                        self.suff_head(pooled.float()).squeeze(-1)
                    ).item()
                suff_scores.append(float(p_insuff))
                suff_labels_eval.append(int(example.get("sufficiency", 0)))

            # Skip undecidable rows for the decision/reason metrics + AR loss.
            if int(example.get("sufficiency", 0)) == 1:
                continue

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
            is_last = idx == last_decided_idx
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

        # Sufficiency-head metrics (needs both classes present in the eval set).
        if self.suff_head is not None and len(set(suff_labels_eval)) > 1:
            all_metrics["sufficiency_auc"] = float(
                roc_auc_score(suff_labels_eval, suff_scores)
            )
            all_metrics["sufficiency_ap"] = float(
                average_precision_score(suff_labels_eval, suff_scores)
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

            # Train sufficiency AUC/AP (dropout-free head scores). Independent of
            # the decision block above — needs both classes in the logging window
            # (unclear prevalence ~4.7%, so a few positives per window; guard).
            if self._train_suff_labels:
                suff_y = np.concatenate(self._train_suff_labels)
                suff_s = np.concatenate(self._train_suff_scores)
                if len(np.unique(suff_y)) > 1:
                    logs["train_sufficiency_auc"] = float(roc_auc_score(suff_y, suff_s))
                    logs["train_sufficiency_ap"] = float(
                        average_precision_score(suff_y, suff_s)
                    )
                self._train_suff_scores = []
                self._train_suff_labels = []

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
