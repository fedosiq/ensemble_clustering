import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import combinations
import bisect
import random
from util import CA, resample

from sklearn import datasets


def mv_consensus(ca: np.array):
    """Majority voting consensus function"""
    dist = 1 - ca

    labels = fcluster(linkage(squareform(dist), 'single'), 0.5, 'distance')
    labels -= 1

    return labels


def mv(X, alg, n_base_partitions=30):
    """Basic Majority Voting"""
    base_partitions = []
    for _ in range(n_base_partitions):
        alg.fit(X)
        if hasattr(alg, 'predict'):
            base_partitions.append(alg.predict(X))
        else:
            base_partitions.append(alg.labels_(X))

    ca = CA(np.array(base_partitions))
    labels = mv_consensus(ca)
    return labels


def mv_pp(X, n_base_partitions=30):
    """Majority Voting with random numbers of clusters in the interval [10, sqrt(n_samples)]"""
    k_list = range(10, int(np.sqrt(len(X))))
    base_partitions = [KMeans(n_clusters=random.choice(k_list), init='random', n_init=1).fit_predict(X) for _ in
                       range(n_base_partitions)]

    ca = CA(np.array(base_partitions))
    labels = mv_consensus(ca)
    return labels


def resampled_mv(X, alg, n_base_partitions=30, resample_proportion=0.8):
    """Majority voting with resampling"""
    N = X.shape[0]
    ca = np.zeros((N, N))
    Is = np.zeros((N, N))

    for h in range(n_base_partitions):
        resampled_indices, resampled_data = resample(X, resample_proportion)
        alg.fit(resampled_data)

        if hasattr(alg, 'predict'):
            Mh = alg.predict(resampled_data)
        else:
            Mh = alg.labels_

        id_clusts = np.argsort(Mh)
        sorted_ = Mh[id_clusts]

        k = len(np.unique(sorted_))

        for i in range(k):  # for each cluster
            ia = bisect.bisect_left(sorted_, i)
            ib = bisect.bisect_right(sorted_, i)
            cluster_indices = id_clusts[ia:ib]
            is_ = resampled_indices[cluster_indices]
            ids_ = np.array(list(combinations(is_, 2))).T

            if ids_.size != 0:
                ca[ids_[0], ids_[1]] += 1

        ids_2 = np.array(list(combinations(resampled_indices, 2))).T
        Is[ids_2[0], ids_2[1]] += 1
    Is += Is.T
    ca = ca / (Is + 1e-8)
    ca += ca.T
    ca += np.eye(N)

    labels = mv_consensus(ca)

    return labels


if __name__ == '__main__':
    X, _ = datasets.make_blobs(200, centers=2, n_features=2, shuffle=False)
    labels = resampled_mv(X, KMeans(n_clusters=5), 10)
    print(labels)
