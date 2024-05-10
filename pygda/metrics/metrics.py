from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def eval_roc_auc(label, score):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
    score : torch.Tensor

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """

    roc_auc = roc_auc_score(y_true=label, y_score=score)

    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc

    return roc_auc


def eval_recall_at_k(label, score, k=None):
    """
    Recall score for top k instances.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``.
    score : torch.Tensor
        Scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        recall. Default: ``None``.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances.
    """

    if k is None:
        k = sum(label)
    recall_at_k = sum(label[score.topk(k).indices]) / sum(label)
    return recall_at_k


def eval_precision_at_k(label, score, k=None):
    if k is None:
        k = sum(label)
    precision_at_k = sum(label[score.topk(k).indices]) / k
    return precision_at_k


def eval_average_precision(label, score):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    ap = average_precision_score(y_true=label, y_score=score)
    return ap


def eval_micro_f1(label, pred):
    """
    Micro-F1 score.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``.
    pred : torch.Tensor
        Predictions in shape of ``(N, )``.

    Returns
    -------
    f1 : float
        Micro-F1 score.
    """

    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    f1 = f1_score(y_true=label, y_pred=pred, average='micro')
    return f1

def eval_macro_f1(label, pred):
    """
    Macro-F1 score.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``.
    pred : torch.Tensor
        Predictions in shape of ``(N, )``.

    Returns
    -------
    f1 : float
        Macro-F1 score.
    """

    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    f1 = f1_score(y_true=label, y_pred=pred, average='macro')
    return f1