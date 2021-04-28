import numpy as np
from sklearn.cluster import AgglomerativeClustering


def get_all_segs(base_clusterings: np.array):
    N, n_base_clusterings = base_clusterings.shape
    n_clusters_orig = np.max(base_clusterings, axis=0) + 1
    C = np.cumsum(n_clusters_orig)
    zero = np.zeros(1)
    bcs = base_clusterings + np.concatenate((zero, C[:-1]))

    n_clusters = n_clusters_orig[-1] + C[-2]
    base_clusterings_segments = np.zeros((n_clusters, N))

    for i in range(n_base_clusterings):
        if i == 0:
            startK = 0
        else:
            startK = C[i - 1]
        endK = C[i] - 1
        search_vec = np.array(list(range(startK, endK + 1)))
        F = np.equal(bcs[:, i].reshape(-1, 1), search_vec)
        base_clusterings_segments[search_vec, :] = F.T

    return bcs, base_clusterings_segments


def compute_ECI(bcs, base_clusterings_segments, theta=0.5):
    M = bcs.shape[1]
    ETs = get_all_clusters_entropy(bcs, base_clusterings_segments)
    ECI = np.exp(-ETs / (theta * M))
    return ECI


def get_all_clusters_entropy(bcs, base_clusterings_segments):
    base_clusterings_segments = base_clusterings_segments.T

    N, n_clusters = base_clusterings_segments.shape
    Es = np.zeros((n_clusters, 1))
    for i in range(n_clusters):
        part_bcs = bcs[base_clusterings_segments[:, i] != 0, :]
        Es[i] = get_cluster_entropy(part_bcs)
    return Es


def get_cluster_entropy(part_bcs):
    E = 0
    for i in range(part_bcs.shape[1]):
        tmp = sorted(part_bcs[:, i])
        u_tmp = np.unique(tmp)

        if len(u_tmp) <= 1:
            continue

        cnts = np.zeros(u_tmp.shape)
        for j in range(len(u_tmp)):
            cnts[j] = np.sum(tmp == u_tmp[j])

        cnts = cnts / sum(cnts)
        E = E - sum(cnts * np.log2(cnts))
    return E


def compute_LWCA(base_clusterings_segments: np.array, ECI, M):
    segments = base_clusterings_segments.T
    N = segments.shape[0]
    LWCA = (segments * np.tile(ECI.T, (N, 1))) @ segments.T / M
    LWCA = LWCA - np.diag(np.diag(LWCA)) + np.eye(N)
    return LWCA


def get_consensus_partition(similarity_matrix: np.array, k):
    X = 1 - similarity_matrix

    model = AgglomerativeClustering(n_clusters=k, linkage="complete", affinity='precomputed').fit(X)
    return model.labels_


def lwea(alg, X, n_base_partitions=20, n_clusters=2, theta=0.5, y=None):
    alg.fit(X)
    base_partitions = []
    if hasattr(alg, 'predict'):
        for _ in range(n_base_partitions):
            alg.fit(X)
            base_partitions.append(alg.predict(X))
    else:
        for _ in range(n_base_partitions):
            alg.fit(X)
            base_partitions.append(alg.labels_(X))

    bcs, segments = get_all_segs(np.array(base_partitions).T)
    ECI = compute_ECI(bcs, segments, theta)
    ca = compute_LWCA(segments, ECI, bcs.shape[1])
    labels = get_consensus_partition(ca, n_clusters)
    return labels


if __name__ == '__main__':
    base_clusterings = np.array([
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 2],
        [0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    ]).T
    bcs, segments = get_all_segs(base_clusterings)
    print(bcs)
    print(segments)
    ECI = compute_ECI(bcs, segments)
    print(ECI)
    ca = compute_LWCA(segments, ECI, bcs.shape[1])
    print(ca)
    print(get_consensus_partition(ca, 3))
