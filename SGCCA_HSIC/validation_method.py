## validation method set up
# F1-score
# MCC
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

def FS_MCC(U, V, W, Label):
    tp, tn, fn, fp = 0, 0, 0, 0

    pred = torch.abs(U) > torch.mean(torch.abs(U))
    C = confusion_matrix(pred, Label)
    tp += C[1][1]
    tn += C[0][0]
    fn += C[1][0]
    fp += C[0][1]

    pred = torch.abs(V) > torch.mean(torch.abs(V))
    C = confusion_matrix(pred, Label)
    tp += C[1][1]
    tn += C[0][0]
    fn += C[1][0]
    fp += C[0][1]

    pred = torch.abs(W) > torch.mean(torch.abs(W))
    C = confusion_matrix(pred, Label)
    tp += C[1][1]
    tn += C[0][0]
    fn += C[1][0]
    fp += C[0][1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * precision * recall / (precision + recall)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return f1, mcc