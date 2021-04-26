import os
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from sklearn import datasets
from algorithms import *

from MultiClustering import Constants
from MultiClustering.RLrfAlgoEx import RLrfrsAlgoEx
from MultiClustering.mab_solvers.UCB_SRSU import UCBsrsu

import warnings

warnings.filterwarnings("ignore")



def run_hpo(X, seed, metric, output_file):
    iterations = 5000
    f = open(file=output_file, mode='a')

    algo_e = RLrfrsAlgoEx(metric, X, seed, 1, expansion=100)
    mab_solver = UCBsrsu(action=algo_e, time_limit=20)

    # Random initialization:
    mab_solver.initialize(f, true_labels=None)

    start = time.time()
    # RUN actual Multi-Arm:
    its = mab_solver.iterate(iterations, f)
    time_iterations = time.time() - start

    print("#PROFILE: time spent in iterations:" + str(time_iterations))

    algo_e = mab_solver.action

    # print(algo_e.best_param)
    print(f"{metric} {algo_e.best_val}")

    return algo_e.best_algo, algo_e.best_param


def call_algo(name, cfg):
    if name == Constants.kmeans_algo:
        cl = KMeans(**cfg)
    elif name == Constants.affinity_algo:
        cl = AffinityPropagation(**cfg)
    # elif name == Constants.mean_shift_algo:
    #     bandwidth = estimate_bandwidth(self.X, quantile=cfg['quantile'])
    #     cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(cfg['bin_seeding']), min_bin_freq=cfg['min_bin_freq'],
    #                    cluster_all=bool(cfg['cluster_all']))
    elif name == Constants.ward_algo:
        linkage = cfg["linkage"]
        aff = ""
        if ("ward" in linkage):
            aff = cfg["affinity_w"]
        else:
            aff = cfg["affinity_a"]
        n_c = cfg["n_clusters"]
        cl = AgglomerativeClustering(n_clusters=n_c, linkage=linkage, affinity=aff)
    elif name == Constants.dbscan_algo:
        cl = DBSCAN(**cfg)
    elif name == Constants.gm_algo:
        cl = GaussianMixture(**cfg)
    elif name == Constants.bgm_algo:
        cl = BayesianGaussianMixture(**cfg)
    return cl


def pretuned_synthetic_test():
    scores = {"k-means": [],
              "lwea": [],
              "lwea_tuned": [],
              "monti": [],
              "monti_tuned": [],
              }
    clusters = range(2, 10)
    features = range(2, 10)
    i = 0

    for k in clusters:
        for n_features in features:
            # for _ in range(10):
            print(f"dataset {i + 1} of {len(clusters) * len(features)}")

            X, y = datasets.make_blobs(50 * k, centers=k, n_features=n_features)
            # try:
            alg, conf = run_hpo(X, 42, Constants.silhouette_metric, "experiment.txt")
            print(alg, conf)
            cl = call_algo(alg, conf)

            kmeans = run_k_means(X, k, y)
            lwea = run_lwea(X, k, y)
            lwea_tuned = run_lwea_tuned(X, cl, k, y)
            monti = run_monti(X, 2, k + 5, 50, 0.8)
            monti_tuned = run_monti_tuned(X, cl, 2, 10, 50, 0.8)

            scores["k-means"].append(kmeans)
            scores["lwea"].append(lwea)
            scores["lwea_tuned"].append(lwea_tuned)
            scores["monti"].append(monti)
            scores["monti_tuned"].append(monti_tuned)
            # except Exception as e:
            #     print(e)
            #     print(f"problem dataset {i + 1}")
            #     pass

            i += 1
            print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv("synt_scores_final.csv", index=False)


def synthetic_test():
    scores = {"k-means": [],
              # "mv_oe": [],
              "mv": [],
              # "lwea": [], "monti": [],
              # "monti_fixed": [], "monti_hierarchical": []
              }
    clusters = range(2, 10)
    features = range(2, 10)
    i = 0

    for k in clusters:
        for n_features in features:
            # for _ in range(10):
            print(f"dataset {i + 1} of {len(clusters) * len(features)}")

            X, _ = datasets.make_blobs(50 * k, centers=k, n_features=n_features)
            try:
                scores["k-means"].append(run_k_means(X, k))
                # scores["mv_oe"].append(run_mv_oe(X))
                scores["mv"].append(run_mv(X))
                # scores["lwea_known_n_clusters"].append(run_lwea_known_n_clusters(X, k, y))
                # scores["monti"].append(run_monti(X, 2, k + 5, 50, 0.8, y))
                # scores["monti_fixed"].append(run_monti(X, k, k + 1, 50, 0.8))
                # scores["monti_hierarchical"].append(run_monti(X, 2, k + 5, 50, 0.8))
            except Exception as e:
                print(f"problem dataset {i + 1}: {e}")
                pass

            i += 1
            print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv("test_mvs.csv", index=False)


def test():
    data_path = "../data/with_class"
    data = []
    fnames = os.listdir(data_path)
    for fname in fnames:
        df = pd.read_csv(f"{data_path}/{fname}")
        n_clusters = len(np.unique(df.iloc[:, -1]))
        data.append((df.iloc[:, :-1].values, n_clusters))

    scores = {"k-means": [],
              # "mv_oe": [],
              "lwea": [], "monti": [], "monti_fixed": []}

    for i, (X, k) in enumerate(data):
        print(f"dataset {i + 1} of {len(data)}")
        scores["k-means"].append(run_k_means(X, k))
        # scores["mv_oe"].append(run_mv_oe(X))
        scores["lwea_known_n_clusters"].append(run_lwea_known_n_clusters(X, k))
        scores["monti"].append(run_monti(X, 2, k + 5, 50, 0.8))
        scores["monti_fixed"].append(run_monti(X, k, k + 1, 50, 0.8))
        print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv("scores1.csv", index=False)


if __name__ == '__main__':
    pretuned_synthetic_test()
    # test()
