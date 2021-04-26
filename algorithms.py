import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import openensembles as oe
from LWEA import LWEA
from monti import ConsensusCluster

from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import calinski_harabaz_score as ch
from metrics import gd41, os_score

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def with_metrics(test, n_runs=5):
    def wrapper(*args):
        cvis = []
        accs = []
        for _ in range(n_runs):
            X, labels, y = test(*args)

            cvis.append([metric(X, labels) for metric in [sil, ch, gd41, os_score]])

            # if y is not None:
            #     acc = accuracy_score(y, labels)
            #     accs.append(acc)
            # else:
            #     acc = None

        averaged_cvis = np.array(cvis).mean(axis=0)
        print(averaged_cvis)
        # return cvis, acc
        return averaged_cvis

    return wrapper


@with_metrics
def run_k_means(X, n_clusters, y=None):
    kmeans = KMeans(n_clusters)
    labels = kmeans.fit_predict(X)
    return X, labels, y


@with_metrics
def run_mv_oe(X, y=None):
    n_features = X.shape[1]
    columns = [f"x{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=columns)
    dataObj = oe.data(df, list(range(n_features)))

    c = oe.cluster(dataObj)
    c_MV_arr = []

    for i in range(100):
        name = f'kmeans_{i}'
        c.cluster('parent', 'kmeans', name, K=15, init='random', n_init=1)
        c_MV_arr.append(c.finish_majority_vote(threshold=0.5))

    final_labels = c_MV_arr[-1].labels['majority_vote'] - 1

    return X, final_labels, y if len(np.unique(final_labels)) > 1 else run_mv_oe(X, y)


@with_metrics
def run_lwea_known_n_clusters(X, n_clusters, y=None):
    labels_ensemble = np.array([KMeans(n_clusters).fit_predict(X) for _ in range(20)])
    bcs, segments = LWEA.get_all_segs(labels_ensemble.T)
    ECI = LWEA.compute_ECI(bcs, segments)
    ca = LWEA.compute_LWCA(segments, ECI, bcs.shape[1])
    labels = LWEA.LWEA(ca, n_clusters)

    return X, labels, y


@with_metrics
def run_lwea(X, y=None):
    labels_ensemble = np.array([KMeans().fit_predict(X) for _ in range(20)])
    bcs, segments = LWEA.get_all_segs(labels_ensemble.T)
    ECI = LWEA.compute_ECI(bcs, segments)
    ca = LWEA.compute_LWCA(segments, ECI, bcs.shape[1])
    labels = LWEA.LWEA(ca, 2)

    return X, labels, y


@with_metrics
def run_lwea_tuned(X, alg, n_clusters, y=None):
    alg.fit(X)
    if hasattr(alg, 'predict'):
        labels_ensemble = np.array([alg.predict(X) for _ in range(20)])
    else:
        labels_ensemble = np.array([alg.labels_ for _ in range(20)])

    bcs, segments = LWEA.get_all_segs(labels_ensemble.T)
    ECI = LWEA.compute_ECI(bcs, segments)
    ca = LWEA.compute_LWCA(segments, ECI, bcs.shape[1])
    labels = LWEA.LWEA(ca, n_clusters)

    return X, labels, y



@with_metrics
def run_monti(X, L, n_clusters, H=10, proportion=0.5, y=None):
    monti = ConsensusCluster(KMeans, L, K=n_clusters, H=H, resample_proportion=proportion)
    monti.fit(X)
    labels = monti.predict()
    return X, labels, y


@with_metrics
def run_monti_hierarchical(X, L, n_clusters, H=10, proportion=0.5, y=None):
    monti = ConsensusCluster(KMeans, L, K=n_clusters, H=H, resample_proportion=proportion)
    monti.fit(X)
    labels = monti.predict_hierarchical()
    return X, labels, y


@with_metrics
# TODO: remove L, K as they make no difference with tuning
def run_monti_tuned(X, alg, L, n_clusters, H=10, proportion=0.5, y=None):
    monti = ConsensusCluster(alg, L, K=n_clusters, H=H, resample_proportion=proportion)
    monti.fit_from_cfg(X)
    labels = monti.predict_from_tuned()
    return X, labels, y



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
    base = [KMeans(n_clusters=n_base_clusters, init='random', n_init=1).fit_predict(X) for _ in
            range(n_base_partitions)]

    ca = CA(np.array(base))
    dist = 1 - ca

    labels = fcluster(linkage(squareform(dist), 'single'), 0.5, 'distance')
    labels -= 1

    return labels


@with_metrics
def run_mv(X, y=None):
    labels = mv(X, 100)
    return X, labels, y
