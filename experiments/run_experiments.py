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
import traceback

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

              # "mv": [],
              # "mv_pp": [],
              "eac_sl": [],
              "eac_al": [],
              "lwea": [],
              "monti": [],

              # "mv_tuned": [],
              "eac_sl_tuned": [],
              "eac_al_tuned": [],
              # "resampled_mv_tuned": [],
              # "lwea_tuned": [],
              "monti_tuned": [],
              }

    for i, (X, k) in enumerate(data):
        print(f"dataset {i + 1} of {len(data)}")

        try:
            alg, conf = run_hpo(X, 42, Constants.silhouette_metric, "../tuning.txt")
            print(alg, conf)
            cl = call_algo(alg, conf)

            k_means = run_k_means(X, k)
            # m_voting = run_mv(X, 20)
            eac_sl = run_eac(X, linkage_method='single')
            eac_al = run_eac(X, linkage_method='average')
            lwea = run_lwea(X)
            monti = run_monti(X, 2, k + 5, 50, 0.8)

            # mv_tuned = run_mv_tuned(X, cl)
            eac_sl_tuned = run_eac(X, alg=cl, linkage_method='single')
            eac_al_tuned = run_eac(X, alg=cl, linkage_method='average')
            # resampled_mv_tuned = run_resampled_mv_tuned(X, cl)
            # lwea_tuned = run_lwea_tuned(X, cl, k)
            monti_tuned = run_monti_tuned(X, cl, 2, 10, 50, 0.8)

            scores["k-means"].append(k_means)

            # scores["mv"].append(m_voting)
            scores["eac_sl"].append(eac_sl)
            scores["eac_al"].append(eac_al)
            scores["lwea"].append(lwea)
            scores["monti"].append(monti)

            # scores["mv_tuned"].append(mv_tuned)
            scores["eac_sl_tuned"].append(eac_sl_tuned)
            scores["eac_al_tuned"].append(eac_al_tuned)
            # scores["resampled_mv_tuned"].append(resampled_mv_tuned)
            # scores["lwea_tuned"].append(lwea_tuned)
            scores["monti_tuned"].append(monti_tuned)

        except Exception as e:
            print(f"problem dataset {i + 1}: {e}")
            traceback.print_exc()
            pass

        print()

    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.sum() / len(scores))
    scores.to_csv(output_fname, index=False)


if __name__ == '__main__':
    real_data_path = "../data/with_class"
    synth_data_path = "../data/synthetic"

    real_data = util.read_data(real_data_path)
    synthetic_data = util.read_data(synth_data_path)

    # test(synthetic_data, "synth_test.csv")
    test(real_data, "real_test.csv")
