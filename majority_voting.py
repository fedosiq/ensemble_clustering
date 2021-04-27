import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster


def CA(base_partitions: np.array):
    N = base_partitions.shape[1]
    ca = np.zeros((N, N))
    for partition in base_partitions:
        for i, l1 in enumerate(partition):
            for j, l2 in enumerate(partition):
                if l1 == l2:
                    ca[i][j] += 1

    return 1 / len(base_partitions) * ca


def mv(X, n_base_partitions=30, n_base_clusters=15):
    base_partitions = [KMeans(n_clusters=n_base_clusters, init='random', n_init=1).fit_predict(X) for _ in
            range(n_base_partitions)]

    ca = CA(np.array(base_partitions))
    dist = 1 - ca

    labels = fcluster(linkage(squareform(dist), 'single'), 0.5, 'distance')
    labels -= 1

    return labels
