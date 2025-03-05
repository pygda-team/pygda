from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def eval_roc_auc(label, score):
    """
    Calculate ROC-AUC score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Ground truth binary labels in shape of (N,)
    score : torch.Tensor
        Predicted scores/probabilities in shape of (N,)

    Returns
    -------
    float
        ROC-AUC score, adjusted to be in [0.5, 1.0] range

    Notes
    -----
    Processing Steps:

    - Convert tensors to numpy arrays
    - Calculate standard ROC-AUC
    - Adjust scores < 0.5 to their complement

    Features:
    
    - CPU computation
    - Score normalization
    - Binary classification focus
    """
    label = label.cpu().numpy()
    score = score.cpu().numpy()
    roc_auc = roc_auc_score(y_true=label, y_score=score)

    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc

    return roc_auc


def eval_recall_at_k(label, score, k=None):
    """
    Calculate Recall@K metric for top-k predictions.

    Parameters
    ----------
    label : torch.Tensor
        Ground truth binary labels in shape of (N,)
    score : torch.Tensor
        Predicted scores/probabilities in shape of (N,)
    k : int, optional
        Number of top instances to consider
        If None, uses number of positive labels

    Returns
    -------
    float
        Recall score for top-k predictions

    Notes
    -----
    Processing Steps:

    - Determine k value (if not provided)
    - Get top-k scoring instances
    - Calculate recall as (true positives) / (total positives)

    Features:
    
    - Flexible k selection
    - Efficient top-k computation
    - Normalized metric
    """
    if k is None:
        k = sum(label)
    recall_at_k = sum(label[score.topk(k).indices]) / sum(label)
    return recall_at_k


def eval_precision_at_k(label, score, k=None):
    """
    Calculate Precision@K metric for top-k predictions.

    Parameters
    ----------
    label : torch.Tensor
        Ground truth binary labels in shape of (N,)
    score : torch.Tensor
        Predicted scores/probabilities in shape of (N,)
    k : int, optional
        Number of top instances to consider
        If None, uses number of positive labels

    Returns
    -------
    float
        Precision score for top-k predictions

    Notes
    -----
    Processing Steps:

    - Determine k value (if not provided)
    - Get top-k scoring instances
    - Calculate precision as (true positives) / k

    Features:
    
    - Flexible k selection
    - Efficient top-k computation
    - Normalized metric
    """
    if k is None:
        k = sum(label)
    precision_at_k = sum(label[score.topk(k).indices]) / k
    return precision_at_k


def eval_average_precision(label, score):
    """
    Calculate Average Precision score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Ground truth binary labels in shape of (N,)
        1 represents outliers, 0 represents normal instances
    score : torch.Tensor
        Predicted outlier scores in shape of (N,)

    Returns
    -------
    float
        Average Precision score

    Notes
    -----
    Processing Steps:

    - Convert tensors to numpy arrays
    - Calculate average precision using scikit-learn
    - Handle binary classification scenario

    Features:
    
    - CPU computation
    - Outlier detection focus
    - Balanced metric
    """
    score = score.cpu().numpy()
    label = label.cpu().numpy()

    ap = average_precision_score(y_true=label, y_score=score)
    return ap


def eval_micro_f1(label, pred):
    """
    Calculate Micro-F1 score for multi-class classification.

    Parameters
    ----------
    label : torch.Tensor
        Ground truth labels in shape of (N,)
    pred : torch.Tensor
        Predicted class labels in shape of (N,)

    Returns
    -------
    float
        Micro-averaged F1 score

    Notes
    -----
    Processing Steps:

    - Convert tensors to numpy arrays
    - Calculate micro-averaged F1 score
    - Handle multi-class scenario

    Features:
    
    - CPU computation
    - Instance-weighted averaging
    - Multi-class support
    """
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    f1 = f1_score(y_true=label, y_pred=pred, average='micro')
    return f1

def eval_macro_f1(label, pred):
    """
    Calculate Macro-F1 score for multi-class classification.

    Parameters
    ----------
    label : torch.Tensor
        Ground truth labels in shape of (N,)
    pred : torch.Tensor
        Predicted class labels in shape of (N,)

    Returns
    -------
    float
        Macro-averaged F1 score

    Notes
    -----
    Processing Steps:

    - Convert tensors to numpy arrays
    - Calculate macro-averaged F1 score
    - Handle multi-class scenario

    Features:
    
    - CPU computation
    - Class-weighted averaging
    - Multi-class support
    """
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    f1 = f1_score(y_true=label, y_pred=pred, average='macro')
    return f1