import os
import time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

import util
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


def test(data, output_fname):
    scores = {"k-means": [],

              "mv": [],
              "lwea": [],
              "monti": [],

              "lwea_tuned": [],
              "monti_tuned": [],
              }

    for i, (X, k) in enumerate(data):
        print(f"dataset {i + 1} of {len(data)}")

        try:
            alg, conf = run_hpo(X, 42, Constants.silhouette_metric, "../tuning.txt")
            print(alg, conf)
            cl = call_algo(alg, conf)

            scores["k-means"].append(run_k_means(X, k))
            scores["mv"].append(run_mv(X))
            scores["lwea"].append(run_lwea(X))
            scores["monti"].append(run_monti(X, 2, k + 5, 50, 0.8))

            scores["lwea_tuned"].append(run_lwea_tuned(X, cl, k))
            scores["monti_tuned"].append(run_monti_tuned(X, cl, 2, 10, 50, 0.8))

        except Exception as e:
            print(f"problem dataset {i + 1}: {e}")
            pass

        print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv(output_fname, index=False)


if __name__ == '__main__':
    data_path = "../data/with_class"
    real_data = []
    fnames = os.listdir(data_path)
    for fname in fnames:
        df = pd.read_csv(f"{data_path}/{fname}")
        n_clusters = len(np.unique(df.iloc[:, -1]))
        real_data.append((df.iloc[:, :-1].values, n_clusters))

    synthetic_data = util.make_synthetic_datasets()

    test(synthetic_data, "synth_test.csv")
    # pretuned_synthetic_test()
    # test()
