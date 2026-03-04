import torch

def gather_yes_no_logprobs(logprobs, tokenizer):
    # logprobs: [batch, vocab_size]
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]
    gather_idx = torch.tensor([no_id, yes_id], device=logprobs.device)

    gathered = logprobs[..., gather_idx] # gathered: [batch, seq_len] or [batch, seq_len, 2] depending on input shape
    return gathered


def find_best_threshold(yes_probs, y_true):
    # yes_probs = yes_no_scores[:,1] / torch.sum(yes_no_scores,1)
    # yes_probs = torch.sigmoid(yes_no_scores[:,1] - yes_no_scores[:,0])
    from sklearn.metrics import f1_score
    import numpy as np

    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.0, 1.01, 0.005):
        y_pred = convert_probs_to_labels(yes_probs, tokenizer=None, threshold=threshold)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

def convert_scores_to_probs(yes_no_scores):
    yes_probs = torch.sigmoid(yes_no_scores[:,1] - yes_no_scores[:,0])
    return yes_probs

def convert_probs_to_labels(yes_probs, tokenizer, threshold=0.5):
    y_pred = (yes_probs >= threshold).to(int)
    return y_pred


def percent_to_review_for_recall(preds_with_conf, y_true, recall_target=0.95):
    """
    Estimate the percent of records (from highest-confidence down) that must be
    reviewed to achieve a given recall for the positive class.

    Args:
        preds_with_conf (list of tuple): sequence of (pred_label, confidence)
            where `confidence` is the model's score/probability for the
            positive class for that record.
        y_true (list-like): true labels (0/1) for each record.
        recall_target (float): desired recall (between 0 and 1), default 0.95.

    Returns:
        float: percentage (0-100) of records that need to be reviewed to
        reach the desired recall. Returns 0.0 if there are no true positives.
    """
    if recall_target <= 0.0:
        return 0.0

    n = len(y_true)
    if n == 0:
        return 0.0

    # Ensure preds_with_conf length matches
    if len(preds_with_conf) != n:
        raise ValueError("Length of preds_with_conf must match length of y_true")

    total_positives = sum(1 for lab in y_true if int(lab) == 1)
    if total_positives == 0:
        return 0.0

    # Extract confidences (assume second element is confidence)
    confidences = [float(item[1]) for item in preds_with_conf]

    # Sort indices by confidence descending
    sorted_indices = sorted(range(n), key=lambda i: confidences[i], reverse=True)

    cum_tp = 0
    for k, idx in enumerate(sorted_indices, start=1):
        if int(y_true[idx]) == 1:
            cum_tp += 1
        if cum_tp / total_positives >= recall_target:
            return (k / n) * 100.0

    # If loop completes, all records are needed
    return 100.0

