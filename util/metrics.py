import numpy as np
from sklearn.metrics import confusion_matrix


def cluster_perform(idxs_users, atk_num, suspicious, score):  # cluster
    actuals = np.zeros_like(idxs_users)  # benign
    actuals[np.where(idxs_users < atk_num)] = 1  # malicious

    predictions = np.zeros_like(idxs_users)
    if len(suspicious) != 0:
        predictions[suspicious] = 1
    # if score == 0:
    #     predictions = 1 - predictions

    res = confusion_matrix(actuals, predictions, labels=[1, 0])
    TP, FN, FP, TN = res[0][0], res[0][1], res[1][0], res[1][1]
    TPR = TP * 1.0 / (TP + FN) if TP + FN > 0 else -1
    TNR = TN * 1.0 / (TN + FP) if TN + FP > 0 else -1
    FPR = FP * 1.0 / (TN + FP) if TN + FP > 0 else -1
    FNR = FN * 1.0 / (TP + FN) if TP + FN > 0 else -1
    detection_acc = (TP + TN) * 1.0 / len(idxs_users)
    precision = TP * 1.0 / (TP + FP) if TP + FP > 0 else -1
    return TPR, TNR, FPR, FNR, detection_acc, precision


def classify_perform(idxs_users, atk_num, suspicious):  # classify
    actuals = np.zeros_like(idxs_users)  # benign
    actuals[np.where(idxs_users < atk_num)] = 1  # malicious

    predictions = np.zeros_like(idxs_users)
    if len(suspicious) != 0:
        predictions[suspicious] = 1
    reverse = 1 - predictions

    if np.sum(reverse == actuals) > np.sum(predictions == actuals):
        score = 0
    else:
        score = 1

    return score


def class_accuracy(actuals, predictions):
    class_acc = {}
    for i, r in enumerate(confusion_matrix(actuals, predictions)):
        if np.sum(r) != 0:
            class_acc['class' + str(i)] = r[i] / np.sum(r) * 100
        else:
            class_acc['class' + str(i)] = -1
    return class_acc
