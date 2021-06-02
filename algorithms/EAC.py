import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import random

from util import CA


def eac(X, alg=None, linkage_method='single', n_base_partitions=30):
    base_partitions = []
    if alg is None:
        k_list = range(10, int(np.sqrt(len(X)))) if int(np.sqrt(len(X))) > 10 else [10]
        base_partitions = [KMeans(n_clusters=random.choice(k_list), init='random', n_init=1).fit_predict(X) for _ in
                           range(n_base_partitions)]
    else:
        for _ in range(n_base_partitions):
            alg.fit(X)
            if hasattr(alg, 'predict'):
                base_partitions.append(alg.predict(X))
            else:
                base_partitions.append(alg.labels_)

    ca = CA(np.array(base_partitions))
    dist = 1 - ca

    l = linkage(squareform(dist), linkage_method)

    n_clusters = np.argmin(np.diff(l[-20:, -2][::-1])) + 2
    labels = fcluster(l, n_clusters, 'maxclust')
    labels -= 1

    return labels


def eac_sl(X, alg=None, n_base_partitions=30):
    return eac(X, alg, 'single', n_base_partitions)


def eac_al(X, alg=None, n_base_partitions=30):
    return eac(X, alg, 'average', n_base_partitions)
