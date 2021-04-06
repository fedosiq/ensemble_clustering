import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import openensembles as oe
from LWEA import LWEA
from monti import ConsensusCluster

from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import calinski_harabaz_score as ch
from sklearn.metrics import accuracy_score
from metrics import gd41, os

from sklearn import datasets


def with_metrics(test, n_runs=10):
    def wrapper(*args):
        cvis = []
        accs = []
        for _ in range(n_runs):
            X, labels, y = test(*args)

            cvis.append([metric(X, labels) for metric in [sil, ch, gd41, os]])

            if y is not None:
                acc = accuracy_score(y, labels)
                accs.append(acc)
            else:
                acc = None

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
def run_mv(X, y=None):
    n_features = X.shape[1]
    columns = [f"x{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=columns)
    dataObj = oe.data(df, list(range(n_features)))

    c = oe.cluster(dataObj)
    c_MV_arr = []

    for i in range(20):
        name = f'kmeans_{i}'
        c.cluster('parent', 'kmeans', name, K=10, init='random', n_init=1)
        c_MV_arr.append(c.finish_majority_vote(threshold=0.5))

    final_labels = c_MV_arr[-1].labels['majority_vote'] - 1

    return X, final_labels, y if len(np.unique(final_labels)) > 1 else run_mv(X, y)


@with_metrics
def run_lwea(X, n_clusters, y=None):
    labels_ensemble = np.array([KMeans(n_clusters=5).fit_predict(X) for _ in range(20)])
    bcs, segments = LWEA.get_all_segs(labels_ensemble.T)
    ECI = LWEA.compute_ECI(bcs, segments)
    ca = LWEA.compute_LWCA(segments, ECI, bcs.shape[1])
    labels = LWEA.LWEA(ca, n_clusters)

    return X, labels, y


@with_metrics
def run_monti(X, n_clusters, y=None):
    monti = ConsensusCluster(KMeans, 2, K=n_clusters, H=2)
    monti.fit(X)
    labels = monti.predict()
    return X, labels, y


if __name__ == '__main__':
    datasets = [(*datasets.make_blobs(200, centers=2, shuffle=False, random_state=42), 2),
                (*datasets.make_blobs(200, centers=3, shuffle=False, random_state=42), 3),
                (*datasets.make_blobs(200, centers=4, shuffle=False, random_state=42), 4),
                (*datasets.make_blobs(200, n_features=3, centers=2, shuffle=False, random_state=12), 2),
                (*datasets.make_blobs(200, n_features=3, centers=3, shuffle=False, random_state=12), 3),
                (*datasets.make_moons(200, shuffle=False, noise=0.03), 2),

                (*datasets.load_iris(True), 3),
                (pd.read_csv("../data/country.csv").values, None, 4),
                (pd.read_csv("../data/wine.csv").values, None, 3)]

    scores = {"k-means": [], "mv": [], "lwea": [], "monti": []}

    for X, y, k in datasets:
        scores["k-means"].append(run_k_means(X, k, y))
        scores["mv"].append(run_mv(X, y))
        scores["lwea"].append(run_lwea(X, k, y))
        scores["monti"].append(run_monti(X, k + 2, y))
        print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv("scores.csv")
