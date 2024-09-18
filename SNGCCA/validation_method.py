## validation method set up
# F1-score
# MCC
import torch
from sklearn.metrics import confusion_matrix, pairwise_distances, silhouette_score
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
    for i in range(len(U)):
        pred = torch.abs(U[i]) > 5e-2
        C = confusion_matrix(pred, Label)
        tp += C[1][1]
        tn += C[0][0]
        fn += C[1][0]
        fp += C[0][1]

        p0 = tp / (tp + fp + 1e-300)
        precision.append(p0)
        r0 = tp / (tp + fn + 1e-300)
        recall.append(r0)
        if tp != 0:
            f1.append(2 * p0 * r0 / (p0 + r0))
        else:
            f1.append(0)
        spe.append(tn / (tn + fp))
        acc.append((tp + tn) / (tp+tn+fp+fn))

        mcc.append((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        sr.append(check_success(pred, k))

    return spe, precision, recall, acc, f1, mcc, sr

def eval_topk(U,Label,k):
    tp, tn, fn, fp = 0, 0, 0, 0
    for i in range(len(U)):

        top_k_indices = torch.topk(abs(U[i].view(-1)), k).indices
        pred = torch.zeros_like(U[i])
        pred[top_k_indices] = 1

        C = confusion_matrix(pred, Label)
        tp += C[1][1]
        tn += C[0][0]
        fn += C[1][0]
        fp += C[0][1]

    precision = tp / (tp + fp + 1e-300)
    recall = tp / (tp + fn + 1e-300)
    if tp != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    spe = tn / (tn + fp)
    acc = (tp + tn) / (tp+tn+fp+fn)

    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return spe, precision, recall, acc, f1, mcc

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
    if torch.any(tensor[:n]):
        # 如果前 n 个元素中有 True，则检查后面元素是否全为 False
        if torch.all(tensor[n:] == False):
            return 1
    return 0