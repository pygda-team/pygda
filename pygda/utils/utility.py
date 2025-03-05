import numpy as np

def logger(epoch=0,
           loss=0,
           source_train_acc=None,
           source_val_acc=None,
           target=None,
           time=None,
           verbose=0,
           train=True):
    """
    Print formatted training/testing progress information.

    Parameters
    ----------
    epoch : int, optional
        Current training epoch. Default: 0
    loss : float or tuple, optional
        Loss value(s) for current epoch. If tuple, contains inner and outer losses.
        Default: 0
    source_train_acc : float, optional
        Source domain training accuracy. Default: None
    source_val_acc : float, optional
        Source domain validation accuracy. Default: None
    target : torch.Tensor, optional
        Target domain predictions/labels. Default: None
    time : float, optional
        Time taken for current epoch. Default: None
    verbose : int, optional
        Verbosity level controlling output detail:

        - 0: No output
        - 1: Basic loss information
        - 2: Add accuracy metrics
        - 3: Add detailed metrics (recall, precision, etc.)

        Default: 0
    train : bool, optional
        Whether in training or testing mode. Default: True

    Notes
    -----
    Output Levels:

    - Basic Output (verbose=1):
       
       * Epoch number (training) or "Test" (testing)
       * Loss values (single or inner/outer)

    - Extended Output (verbose=2):
       
       * Basic output
       * Source domain accuracy
       * Target domain accuracy
       * Timing information

    - Detailed Output (verbose=3):
       
       * Extended output
       * Recall at k
       * Precision at k
       * Average precision
       * F1 score
       * Contamination metrics

    Features:
    
    - Multi-level verbosity
    - Flexible metric display
    - Progress tracking
    - Performance monitoring

    Format:

    - Epoch XXXX: Loss X.XXXX | Source Acc X.XXXX | Target Acc X.XXXX | Metrics ... | Time X.XX
    """
    if verbose > 0:
        if train:
            print("Epoch {:04d}: ".format(epoch), end='')
        else:
            print("Test: ", end='')

        if isinstance(loss, tuple):
            print("Loss I {:.4f} | Loss O {:.4f} | "
                      .format(loss[0], loss[1]), end='')
        else:
            print("loss {:.4f}, ".format(loss), end='')

        if verbose > 1:
            if source_train_acc is not None:
                print("source acc {:.4f}, ".format(source_train_acc), end='')

            if target is not None:
                print("target acc {:.4f}, ".format(target), end='')

            if verbose > 2:
                if target is not None:
                    pos_size = target.nonzero().size(0)
                    rec = eval_recall_at_k(target, score, pos_size)
                    pre = eval_precision_at_k(target, score, pos_size)
                    ap = eval_average_precision(target, score)

                    contamination = sum(target) / len(target)
                    threshold = np.percentile(score,
                                              100 * (1 - contamination))
                    pred = (score > threshold).long()
                    f1 = eval_f1(target, pred)

                    print(" | Recall {:.4f} | Precision {:.4f} "
                          "| AP {:.4f} | F1 {:.4f}"
                          .format(rec, pre, ap, f1), end='')

            if time is not None:
                print("time {:.2f}".format(time), end='')

        print()