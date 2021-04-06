import numpy as np
from util import *
import sys

def dunn(X: np.array, labels: np.array):
    ks = np.unique(labels)
    k_list = [X[labels == k] for k in ks]

    deltas = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas) / np.max(big_deltas)
    return di


def gd41(X, labels):
    n_clusters = len(np.unique(labels))
    centroids = cluster_centroid(X, labels, n_clusters)

    rows, colums = X.shape
    minimum_dif_c = sys.float_info.max
    maximum_same_c = sys.float_info.min
    centres_l = [[0.0] * n_clusters] * n_clusters
    centers = np.array(centres_l)
    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            centers[i][j] = euclidian_dist(centroids[i], centroids[j])
            centers[j][i] = euclidian_dist(centroids[i], centroids[j])

    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                dist = centers[labels[i]][labels[j]]
                minimum_dif_c = min(dist, minimum_dif_c)
            else:
                dist = euclidian_dist(X[i], X[j])
                maximum_same_c = max(dist, maximum_same_c)
    return minimum_dif_c / maximum_same_c


def os(X, labels):
    n_clusters = len(np.unique(labels))
    centroids = cluster_centroid(X, labels, n_clusters)
    cluster_sizes = count_cluster_sizes(labels, n_clusters)

    numerator = 0.0
    for k in range(0, n_clusters):
        for i in range(0, len(labels)):
            if labels[i] != k: continue
            numerator += ov(X, labels, X[i], k, cluster_sizes[k])

    denominator = 0.0
    for k in range(0, n_clusters):
        l = []
        for i in range(0, len(labels)):
            if labels[i] != k:
                continue
            l.append(euclidian_dist(X[i], centroids[k]))

        # get sum of 0.1*|Ck| largest elements
        acc = 0.0
        max_n = heapq.nlargest(int(math.ceil(0.1 * cluster_sizes[k])), l)
        for i in range(0, len(max_n)):
            acc += max_n[i]

        denominator += acc * 10.0 / cluster_sizes[k]

    return numerator / denominator
