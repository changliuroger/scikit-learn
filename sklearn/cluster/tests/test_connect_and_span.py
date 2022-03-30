import numpy as np
from sklearn.cluster import ConnectedComponentsClustering
from sklearn.cluster import SpanTreeConnectedComponentsClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import rand_score

def test_connected():
    centers = [[1,1], [-1, -1], [1, -1]]

    X, labels_true = make_blobs(n_samples=3000, random_state=170, cluster_std=0.7)
    connected = ConnectedComponentsClustering(threshold = 0.275, metric = "euclidean", n_jobs = -1)
    connected.fit(X)
    z = rand_score(labels_true = labels_true, labels_pred = connected.fit_predict(X))

    assert 0.95 < z <= 1.0

def test_span_tree():
    centers = [[1,1], [-1, -1], [1, -1]]

    X, labels_true = make_blobs(n_samples=3000, random_state=170, cluster_std=0.7)
    span = SpanTreeConnectedComponentsClustering(n_clusters = 3, metric = "euclidean", n_jobs = -1)
    span.fit(X)
    z = rand_score(labels_true = labels_true, labels_pred = span.fit_predict(X))

    assert 0.95 < z <= 1.0
