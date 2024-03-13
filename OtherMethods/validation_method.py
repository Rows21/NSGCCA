## validation method set up
# F1-score
# MCC
import torch
from sklearn.metrics import confusion_matrix, pairwise_distances
import numpy as np

def FS_MCC(U, Label):
    tp, tn, fn, fp = 0, 0, 0, 0
    for i in range(len(U)):
        pred = torch.abs(U[i]) > torch.mean(torch.abs(U[i]))
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

    acc = (tp + tn) / (tp+tn+fp+fn)

    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return acc, f1, mcc

def swiss_score(X, labels):
    # 计算样本间的欧氏距离
    distances = pairwise_distances(X, metric='euclidean')
    
    n = len(X)
    s = 0
    for i in range(n):
        a_i = np.mean(distances[i, labels == labels[i]])
        b_i = np.mean(distances[i, labels != labels[i]])
        s += (b_i - a_i) / max(a_i, b_i)
    
    swiss_score = s / n
    return swiss_score

def get_davies_bouldin(X, labels):

    n_clusters = np.unique(labels).shape[0]
    centroids = np.zeros((n_clusters, len(X[0])), dtype=float)
    s_i = np.zeros(n_clusters)
    for k in range(n_clusters):
        m = k+1  # 遍历每一个簇
        x_in_cluster = X[labels == m]  # 取当前簇中的所有样本
        centroids[k] = np.mean(x_in_cluster, axis=1)  # 计算当前簇的簇中心
        s_i[k] = pairwise_distances(x_in_cluster, [centroids[k]]).mean()  
    centroid_distances = pairwise_distances(centroids)  # [K,K]
    combined_s_i_j = s_i[:, None] + s_i  # [K,k]
    centroid_distances[centroid_distances == 0] = np.inf
    scores = np.max(combined_s_i_j / centroid_distances, axis=1)
    return np.mean(scores)

def ch_score(X, labels):
    # 计算每个簇的中心点
    cluster_centers = []
    for label in set(labels):
        cluster_centers.append(np.mean(X[labels == label], axis=0))
    
    # 计算全局中心点
    global_center = np.mean(X, axis=0)
    
    # 计算簇内平均距离
    n_clusters = len(cluster_centers)
    intra_cluster_distances = []
    for i in range(n_clusters):
        distances = pairwise_distances(X[labels == i], [cluster_centers[i]], metric='euclidean')
        intra_cluster_distances.append(np.mean(distances))
    
    # 计算簇间平均距离
    inter_cluster_distances = pairwise_distances(cluster_centers, [global_center], metric='euclidean')
    inter_cluster_distance = np.mean(inter_cluster_distances)
    
    # 计算 CH Score
    ch_score = (inter_cluster_distance * (len(X) - n_clusters)) / (np.sum(intra_cluster_distances) * (n_clusters - 1))
    return ch_score

