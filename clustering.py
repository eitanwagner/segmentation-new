
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def diag(n, k):
    """

    :param n: dimension of matrix
    :param k: "width" of diagonal
    :return:
    """
    return 1 * (np.abs(np.arange(n)[:, np.newaxis] - np.arange(n)) <= k)


def cluster(X, k=3, use_precomputed=False):
    if use_precomputed:
        clustering = AgglomerativeClustering(n_clusters=len(X) // 50, connectivity=diag(n=len(X), k=3),
                                             affinity="precomputed", linkage="average").fit(X)
    else:
        clustering = AgglomerativeClustering(n_clusters=len(X) // 50, connectivity=diag(n=len(X), k=3)).fit(X)
    ends = np.append(np.nonzero(clustering.labels_[1:] != clustering.labels_[:-1])[0] + 1, len(clustering.labels_))
    starts = np.insert(ends[:-1], 0, 0)
    ends = ends.tolist() + [len(X)]
    return list(zip(starts, ends))
