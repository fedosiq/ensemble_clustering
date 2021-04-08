import numpy as np
import scipy.stats
from sklearn.cluster import AgglomerativeClustering


def get_cluster(labels: np.array, m, n) -> np.array:
    """
    Returns indices for cluster n in m-th partition
    :param labels: matrix of size N x M (labels[i, j] = label of i-th data point in j-th partition)
    :param n: number of cluster
    :param m: number of partition
    :return: indices for cluster n in m-th partition
    """
    partition = labels[m, :]
    return np.where(partition == n)[0]


def p(c1: np.array, c2: np.array) -> float:
    """
    Calculates the intersection of two clusters
    :param c1: indices of objects clustered as c1
    :param c2: indices of objects clustered as c1
    :return: ratio of shared objects in two clusters
    """
    return len(set(c1).intersection(set(c2))) / len(c1)


def entropy(lst):
    """
    Alias for scipy entropy calculation
    """
    return scipy.stats.entropy(lst, base=2)


def partition_entropy(labels, m_partition, n_cluster, target_partition):
    """
    Calculates uncertainty (entropy) of given cluster w.r.t. the given target_partition
    :param m_partition:
    :param n_cluster:
    :param target_partition:
    :param labels:
    :return:
    """
    probs = [p(get_cluster(labels, m_partition, n_cluster), get_cluster(labels, target_partition, i))
             for i in range(len(np.unique(labels[target_partition, :])))]
    return entropy(probs)


def cluster_uncertainty(labels, m_partition, n_cluster):
    """
    Calculates cluster uncertainty w.r.t the whole ensemble of partitions
    :param labels:
    :param m_partition:
    :param n_cluster:
    :return:
    """
    entropies = [partition_entropy(labels, m_partition, n_cluster, m)
                 for m in range(len(labels))]
    return sum(entropies)


def eci(labels, m_partition, n_cluster, theta=0.5) -> np.float32:
    H = cluster_uncertainty(labels, m_partition, n_cluster)
    return np.exp(-H / (theta * len(labels)))


def CA(labels: np.array):
    N = labels.shape[1]
    ca = np.zeros((N, N))
    for partition in labels:
        for i, l1 in enumerate(partition):
            for j, l2 in enumerate(partition):
                if l1 == l2:
                    ca[i][j] += 1

    return 1 / len(labels) * ca


def LWCA(labels: np.array):
    N = labels.shape[1]
    ca = np.zeros((N, N))
    for m, partition in enumerate(labels):
        for i, l1 in enumerate(partition):
            for j, l2 in enumerate(partition):
                if l1 == l2:
                    ca[i][j] += 1 * eci(labels, m, l1)

    return 1 / len(labels) * ca


def LWEA(labels: np.array, k):
    similarity_matrix = LWCA(labels)
    X = np.ones(similarity_matrix.shape) - similarity_matrix

    model = AgglomerativeClustering(n_clusters=k).fit(X)
    return model.labels_


if __name__ == '__main__':
    labels = np.array([
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 2],
        [0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    ])

    c00 = get_cluster(labels, m=0, n=0)
    c01 = get_cluster(labels, 0, 1)
    print(partition_entropy(labels, 0, n_cluster=0, target_partition=1))
    print(cluster_uncertainty(labels, 0, 1))

    for i in range(3):
        print()
        for j in range(3):
            H = cluster_uncertainty(labels, i, j)
            print(H, eci(labels, i, j))

    ca = LWCA(labels)
    print("diag", np.diag(ca))
    print(ca)
    print(LWEA(labels, 3))
