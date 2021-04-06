import numpy as np
import math
import heapq


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
