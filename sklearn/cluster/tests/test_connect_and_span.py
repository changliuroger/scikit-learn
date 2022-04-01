import numpy as np
from numpy.testing import assert_equal
from sklearn.cluster import ConnectedComponentsClustering
from sklearn.cluster import SpanTreeConnectedComponentsClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import rand_score

#Issue 21570 testing
def test_connected():
    centers = [[1,1], [-1, -1], [1, -1]]

    X, labels_true = make_blobs(n_samples=3000, random_state=170, cluster_std=0.7)
    connected = ConnectedComponentsClustering(threshold = 0.275, metric = "euclidean", n_jobs = -1)
    connected.fit(X)
    y = rand_score(labels_true = labels_true, labels_pred = connected.fit_predict(X))

    assert_equal(0.95 < y <= 1.0, True)

def test_span_tree():
    centers = [[1,1], [-1, -1], [1, -1]]

    X, labels_true = make_blobs(n_samples=3000, random_state=170, cluster_std=0.7)
    span = SpanTreeConnectedComponentsClustering(n_clusters = 3, metric = "euclidean", n_jobs = -1)
    span.fit(X)
    y = rand_score(labels_true = labels_true, labels_pred = span.fit_predict(X))

    assert_equal(0.95 < y <= 1.0, True)
