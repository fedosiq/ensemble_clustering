import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import openensembles as oe
from LWEA import LWEA
from monti import ConsensusCluster

from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import calinski_harabaz_score as ch
from metrics import gd41, os_score

from sklearn import datasets


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
    labels_ensemble = np.array([KMeans(n_clusters).fit_predict(X) for _ in range(20)])
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


def synthetic_test():
    scores = {"k-means": [], "mv": [], "lwea": [], "monti": [],
              "monti_fixed": [], "monti_hierarchical": []
              }
    clusters = range(2, 10)
    features = range(2, 10)
    i = 0

    for k in clusters:
        for n_features in features:
            # for _ in range(10):
            print(f"dataset {i + 1} of {len(clusters) * len(features)}")

            X, y = datasets.make_blobs(50 * k, centers=k, n_features=n_features)
            try:
                scores["k-means"].append(run_k_means(X, k, y))
                scores["mv"].append(run_mv(X, y))
                scores["lwea"].append(run_lwea(X, k, y))
                scores["monti"].append(run_monti(X, 2, k + 5, 50, 0.8, y))
                scores["monti_fixed"].append(run_monti(X, k, k + 1, 50, 0.8))
                scores["monti_hierarchical"].append(run_monti(X, 2, k + 5, 50, 0.8))
            except:
                print(f"problem dataset {i + 1}")
                pass

            i += 1
            print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv("synt_scores1.csv", index=False)


def test():
    data_path = "../data/with_class"
    data = []
    fnames = os.listdir(data_path)
    for fname in fnames:
        df = pd.read_csv(f"{data_path}/{fname}")
        n_clusters = len(np.unique(df.iloc[:, -1]))
        data.append((df.iloc[:, :-1].values, n_clusters))

    scores = {"k-means": [],
              # "mv": [],
              "lwea": [], "monti": [], "monti_fixed": []}

    for i, (X, k) in enumerate(data):
        print(f"dataset {i + 1} of {len(data)}")
        scores["k-means"].append(run_k_means(X, k))
        # scores["mv"].append(run_mv(X))
        scores["lwea"].append(run_lwea(X, k))
        scores["monti"].append(run_monti(X, 2, k + 5, 50, 0.8))
        scores["monti_fixed"].append(run_monti(X, k, k + 1, 50, 0.8))
        print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv("scores1.csv", index=False)


if __name__ == '__main__':
    synthetic_test()
    # test()
