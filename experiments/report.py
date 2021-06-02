import pandas as pd
import numpy as np
import ast
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys


def process(row):
    row = str.replace(row, '[', '')
    row = str.replace(row, ']', '')
    lst = row.split()
    return np.array(lst).astype(np.float)


def parse_df(df):
    for column in df.columns:
        df[column] = df[column].apply(process)
    return df


metrics = ['Silhouette', 'Calinski-Harabasz', 'gD41', 'OS']


def transform(df, cols=metrics):
    df = parse_df(df)
    means = df.sum() / len(df)

    res = pd.DataFrame(np.stack(means), columns=cols)
    res.index = means.index
    return res


def transform2(df, cols=metrics):
    df = parse_df(df)

    res = pd.DataFrame()
    for alg in df.columns:
        tmp = pd.DataFrame(np.stack(df[alg]), columns=cols)
        tmp = tmp.apply(lambda col: f"{col.mean():.3f} ± {col.std():.3f}")
        res = res.append(pd.DataFrame(tmp).transpose())
    res.index = df.columns
    return res


def save_boxplots(df, path):
    # letters = ['a', 'b', 'c']
    for m in metrics:
        sils = get_metric_stats(df, m)
        plt.figure(figsize=(9, 5))
        plt.title(f"{m} index")
        sils.boxplot(rot=12)
        plt.savefig(f"{path}{m}", dpi=300)
        # plt.show()


def get_metric_stats(df, metric):
    metric_values = pd.DataFrame()

    for alg in df.columns:
        metric_values[alg] = df[alg].apply(lambda a: a[metrics.index(metric)])
    return metric_values


def make_report(exp_path):
    files = os.listdir(exp_path)
    for f in files:
        name, ext = os.path.splitext(f)
        if ext == '.csv':
            print(f'reporting {f}')
            df = pd.read_csv(f"{exp_path}/{f}")
            res = transform2(df, metrics)
            res.to_csv(f'{exp_path}/{name}_means.csv')
            save_boxplots(df, f"{exp_path}/{name}_means_")


if __name__ == '__main__':
    exp_path = sys.argv[1]
    print(exp_path)
    make_report(exp_path)
