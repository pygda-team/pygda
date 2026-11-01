"""Open-set metrics used by the official UAGA experiments."""
import numpy as np


def evaluate_uaga(labels, predictions, unknown_scores, num_classes):
    """Return OS, OS*, UNK_AUC and HS as fractions in [0, 1].

    num_classes is the known-class count K. Ground truth labels >=K map to
    unknown label K. OS/OS* average recalls over observed classes. HS combines
    micro accuracy on known nodes with unknown recall, following the author
    code. When both HS components are zero, return zero instead of NaN.
    Both known and unknown nodes must be present for unknown-vs-known ROC AUC.
    """
    from sklearn.metrics import roc_auc_score
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError('num_classes must be a positive known-class count.')
    y, pred, score = (np.asarray(v) for v in (labels, predictions, unknown_scores))
    if y.ndim != 1 or pred.shape != y.shape or score.shape != y.shape or len(y) == 0:
        raise ValueError('Labels, predictions and scores must be matching nonempty vectors.')
    if (not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer)
            or np.any(y < 0) or np.any(pred < 0) or np.any(pred > num_classes)):
        raise ValueError('Labels must be nonnegative integers; predictions must be in 0..K.')
    if not np.all(np.isfinite(score)) or np.any((score < 0) | (score > 1)):
        raise ValueError('Unknown scores must be finite probabilities.')
    y = np.minimum(y, num_classes)
    unknown = y == num_classes
    if not unknown.any() or unknown.all():
        raise ValueError('Open-set AUC requires both known and unknown target nodes.')
    classes = np.unique(y)
    recalls = [float(np.mean(pred[y == k] == k)) for k in classes]
    known_accuracy = float(np.mean(pred[~unknown] == y[~unknown]))
    unknown_recall = recalls[-1]
    denominator = known_accuracy + unknown_recall
    return {'OS': float(np.mean(recalls)), 'OS*': float(np.mean(recalls[:-1])),
            'UNK_AUC': float(roc_auc_score(unknown, score)),
            'HS': 2 * known_accuracy * unknown_recall / denominator if denominator else 0.0}
