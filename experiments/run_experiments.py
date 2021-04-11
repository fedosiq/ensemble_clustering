import os
import pandas as pd
import numpy as np

from sklearn import datasets
from algorithms import *


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
                scores["lwea_known_n_clusters"].append(run_lwea_known_n_clusters(X, k, y))
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
        scores["lwea_known_n_clusters"].append(run_lwea_known_n_clusters(X, k))
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
