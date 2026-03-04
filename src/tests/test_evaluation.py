from utils.evaluation import percent_to_review_for_recall
import pytest


def test_percent_basic():
    preds = [(0, 0.9), (0, 0.2), (0, 0.8), (0, 0.1)]
    y_true = [1, 0, 1, 0]
    pct = percent_to_review_for_recall(preds, y_true, recall_target=0.95)
    assert pct == 50.0


def test_percent_all_needed():
    preds = [(0, 0.1), (0, 0.2), (0, 0.3), (0, 0.4)]
    y_true = [1, 1, 0, 0]
    pct = percent_to_review_for_recall(preds, y_true, recall_target=0.95)
    assert pct == 100.0


def test_no_positives():
    preds = [(0, 0.9), (0, 0.8)]
    y_true = [0, 0]
    pct = percent_to_review_for_recall(preds, y_true, recall_target=0.95)
    assert pct == 0.0


def test_length_mismatch():
    preds = [(0, 0.9)]
    y_true = [1, 0]
    with pytest.raises(ValueError):
        percent_to_review_for_recall(preds, y_true)
