import os
import numpy as np
import pandas as pd
import math
import heapq
from sklearn import datasets


def CA(base_partitions: np.array) -> np.array:
    """Calculates co-association matrix from ensemble of partitions"""
    N = base_partitions.shape[1]
    ca = np.zeros((N, N))
    for partition in base_partitions:
        for i, l1 in enumerate(partition):
            for j, l2 in enumerate(partition):
                if l1 == l2:
                    ca[i][j] += 1

    return 1 / len(base_partitions) * ca


def resample(data, proportion):
    """Returns resampled data with indices"""
    resampled_indices = np.random.choice(
        range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False
    )
    return resampled_indices, data[resampled_indices, :]


def cluster_centroid(X, labels, n_clusters):
    rows, colums = X.shape
    center = [[0.0] * colums] * n_clusters
    centroid = np.array(center)
    num_points = np.array([0] * n_clusters)
    for i in range(0, rows):
        c = labels[i]
        num_points[c] += 1

    for i in range(0, rows):
        c = labels[i]
        for j in range(0, colums):
            centroid[c][j] += X[i][j]
            centroid[c][j] %= num_points[c]

    for i in range(0, n_clusters):
        for j in range(0, colums):
            if num_points[i] == 0:
                continue
            centroid[i][j] /= num_points[i]
    return centroid


def count_cluster_sizes(labels, n_clusters):
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    return point_in_c


def euclidian_dist(x1, x2):
    sum = 0.0
    for i in range(0, len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return math.sqrt(sum)


def a(X, labels, x_i, cluster_k_index, cluster_k_size):
    acc = 0.0
    for j in range(0, len(labels)):
        if labels[j] != cluster_k_index: continue
        acc += euclidian_dist(x_i, X[j])
    return acc / cluster_k_size


def b(X, labels, x_i, cluster_k_index, cluster_k_size):
    dists = []
    for j in range(0, len(labels)):
        if (labels[j] != cluster_k_index):
            dists.append(euclidian_dist(x_i, X[j]))

    acc = 0.0
    min_n = heapq.nsmallest(cluster_k_size, dists)
    for i in range(0, len(min_n)):
        acc += min_n[i]
    return acc / cluster_k_size


def ov(X, labels, x_i, cluster_k_index, cluster_k_size):
    a_s = a(X, labels, x_i, cluster_k_index, cluster_k_size)
    b_s = b(X, labels, x_i, cluster_k_index, cluster_k_size)

    if b_s == 0:
        b_s = 0.0000000000001
    if (b_s - a_s) / (b_s + a_s) < 0.4:
        return a_s / b_s
    else:
        return 0


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)]) * 10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)


def make_synthetic_datasets():
    clusters = range(2, 10)
    features = range(2, 10)

    data = []
    for k in clusters:
        for n_features in features:
            X, _ = datasets.make_blobs(50 * k, centers=k, n_features=n_features)
            data.append((X, k))
    return data


def read_data(path):
    """Returns list of (dataset, n_clusters)"""
    data = []
    fnames = os.listdir(path)
    for fname in fnames:
        df = pd.read_csv(f"{path}/{fname}")
        n_clusters = len(np.unique(df.iloc[:, -1]))
        data.append((df.iloc[:, :-1].values, n_clusters))
    return data
