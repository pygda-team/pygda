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
    Logger.

    Parameters
    ----------
    epoch : int, optional
        The current epoch.
    loss : float, optional
        The current epoch loss value.
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