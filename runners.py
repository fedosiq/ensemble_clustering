import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import openensembles as oe
from algorithms import LWEA
from algorithms.monti import ConsensusCluster
from algorithms.majority_voting import mv, mv_pp, resampled_mv
from algorithms.EAC import eac

from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import calinski_harabaz_score as ch
from metrics import gd41, os_score


def with_metrics(test, n_runs=5):
    def wrapper(*args, **kwargs):
        cvis = []
        accs = []
        for _ in range(n_runs):
            X, labels, y = test(*args, **kwargs)

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
    labels = KMeans(n_clusters).fit_predict(X)
    return X, labels, y


@with_metrics
def run_mv(X, n_base_partitions=100, y=None):
    labels = mv(X, KMeans(n_clusters=int(np.sqrt(len(X))), init='random', n_init=1), n_base_partitions)
    return X, labels, y


@with_metrics
def run_mv_pp(X, n_base_partitions=100, y=None):
    labels = mv_pp(X, n_base_partitions)
    return X, labels, y


@with_metrics
def run_eac(X, alg=None, linkage_method='single', n_base_partitions=100, y=None):
    labels = eac(X, alg, linkage_method=linkage_method, n_base_partitions=n_base_partitions)
    return X, labels, y


@with_metrics
def run_mv_tuned(X, alg, n_base_partitions=100, y=None):
    labels = mv(X, alg, n_base_partitions)
    return X, labels, y


@with_metrics
def run_resampled_mv_tuned(X, alg, n_base_partitions=100, resample_proportion=0.8, y=None):
    labels = resampled_mv(X, alg, n_base_partitions, resample_proportion)
    return X, labels, y


@with_metrics
def run_mv_oe(X, y=None):
    """Deprecated"""
    print("a")
    n_features = X.shape[1]
    columns = [f"x{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=columns)
    dataObj = oe.data(df, list(range(n_features)))

    c = oe.cluster(dataObj)
    c_MV_arr = []

    for i in range(30):
        name = f'kmeans_{i}'
        c.cluster('parent', 'kmeans', name, K=15, init='random', n_init=1)
        c_MV_arr.append(c.finish_majority_vote(threshold=0.5))

    final_labels = c_MV_arr[-1].labels['majority_vote'] - 1
    print(len(np.unique(final_labels)))

    return X, final_labels, y if len(np.unique(final_labels)) > 1 else run_mv_oe(X, y)


@with_metrics
def run_lwea_known_n_clusters(X, n_clusters, y=None):
    labels = LWEA.lwea(KMeans(n_clusters), X, n_clusters=n_clusters)
    return X, labels, y


@with_metrics
def run_lwea(X, y=None):
    labels = LWEA.lwea(KMeans(), X)
    return X, labels, y


@with_metrics
def run_lwea_tuned(X, alg, n_clusters, y=None):
    labels = LWEA.lwea(alg, X, n_clusters=n_clusters)
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
def run_monti_tuned(X, alg, L, n_clusters, H=10, proportion=0.5, y=None):
    monti = ConsensusCluster(alg, L, K=n_clusters, H=H, resample_proportion=proportion)
    monti.fit_from_cfg(X)
    labels = monti.predict_from_tuned()
    return X, labels, y
