## validation method set up
# F1-score
# MCC
#import torch
from sklearn.metrics import confusion_matrix, pairwise_distances, f1_score, precision_score, recall_score, matthews_corrcoef
import numpy as np

def eval(U, Label, k):
    spe = []
    precision = []
    recall = []
    acc = []
    f1 = []
    mcc = []
    sr = []
    tp, tn, fn, fp = 0, 0, 0, 0
    y_true = None
    y_pred = None
    for i in range(len(U)):
        pred = (abs(U[i]) > 5e-2).astype(int)
        label = Label.astype(int)
        label = np.concatenate(([1], label[k:]))
        sr.append(check_success(abs(U[i]) > 5e-2, k))
        if np.any(pred[:k]):
            pred = np.concatenate(([1], pred[k:].reshape(-1)))
        else:
            pred = np.concatenate(([0], pred[k:].reshape(-1)))
        if y_true is None:
            y_true = label
            y_pred = pred
        else:
            y_true = np.concatenate((y_true, label))
            y_pred = np.concatenate((y_pred, pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    spe = tn / (tn + fp + 1e-300)

    mcc = matthews_corrcoef(y_true, y_pred)
    sr = (np.stack(sr).sum() == 3) & (fp == 0)

    return spe, precision, recall, f1, mcc, sr

def eval_plot(U, Label, k):
    spe = []
    precision = []
    recall = []
    acc = []
    f1 = []
    mcc = []
    sr = []
    tp, tn, fn, fp = 0, 0, 0, 0
    for i in range(len(U)):
        pred = abs(U[i]) > 5e-2
        C = confusion_matrix(pred, Label)
        tp += C[1][1]
        tn += C[0][0]
        fn += C[1][0]
        fp += C[0][1]

    p0 = tp / (tp + fp + 1e-300)
    precision = p0
    r0 = tp / (tp + fn + 1e-300)
    recall = r0
    if tp != 0:
            f1=2 * p0 * r0 / (p0 + r0 + 1e-300)
    else:
            f1 = 0
    spe = tn / (tn + fp + 1e-300)
    acc = (tp + tn) / (tp+tn+fp+fn + 1e-300)

    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))+ 1e-300)
    sr = check_success(pred, k)

    return spe, precision, recall, acc, f1, mcc, sr


def swiss_score(X, labels):
    # Cal Euclidean Distance
    distances = pairwise_distances(X, metric='euclidean')
    
    n = len(X)
    s = 0
    for i in range(n):
        a_i = np.mean(distances[i, labels == labels[i]])
        b_i = np.mean(distances[i, labels != labels[i]])
        s += (b_i - a_i) / max(a_i, b_i)
    
    swiss_score = s / n
    return swiss_score

def db_score(X, labels):
    # 计算每个簇的中心点
    cluster_centers = []
    for label in set(labels):
        cluster_centers.append(np.mean(X[labels == label], axis=0))
    
    # 计算簇内平均距离
    n_clusters = len(cluster_centers)
    intra_cluster_distances = []
    for i in range(n_clusters):
        distances = pairwise_distances(X[labels == i], [cluster_centers[i]], metric='euclidean')
        intra_cluster_distances.append(np.mean(distances))
    
    # 计算簇间距离
    cluster_distances = pairwise_distances(cluster_centers, metric='euclidean')
    
    db_score = np.mean((intra_cluster_distances[:, np.newaxis] + intra_cluster_distances) / cluster_distances)
    return db_score

def check_success(tensor, n):
    # 检查前 n 个元素中是否有 True
    if np.any(tensor[:n]):
        # 如果前 n 个元素中有 True，则检查后面元素是否全为 False
        if np.all(tensor[n:] == False):
            return 1
    return 0