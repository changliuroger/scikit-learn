"""Graph Based Clustering.

source: https://github.com/dayyass/graph-based-clustering
"""

import warnings

import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClusterMixin, TransformerMixin
from ..metrics.pairwise import euclidean_distances
from ..metrics.pairwise import _euclidean_distances
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_limits
from ..utils.fixes import threadpool_info
from ..utils.sparsefuncs import mean_variance_axis
from ..utils import check_array
from ..utils import check_random_state
from ..utils import deprecated
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..exceptions import ConvergenceWarning
from typing import Callable, Optional, Tuple, Union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from ..metrics import pairwise_distances

def _check_matrix(a: np.ndarray) -> bool:
    """Check if np.ndarray is a matrix.

    Parameters
    ----------
    a (np.ndarray): np.ndarray to check.

    Returns
    ----------
    bool: np.ndarray is a matrix.
    """
    return a.ndim == 2

def _check_matrix_is_square(a: np.ndarray) -> bool:
    """Check if a matrix is square.

    Parameters
    ----------
    a (np.ndarray): A matrix to check.

    Returns
    ----------
    bool: A matrix is square.
    """
    M, N = a.shape

    return M == N

def _check_square_matrix_is_symmetric(a: np.ndarray) -> bool:
    """Check if a square matrix is symmetric.

    Parameters
    ----------
    a : np.ndarray
        A square matrix to check.

    rtol : float, optional, defaults = 1e-05
        The relative tolerance parameter.

    atol : float, optional, defaults = 1e-08
        The absolute tolerance parameter.

    Returns
    ----------
    bool: A square matrix is symmetric.
    """
    return np.allclose(a, a.T)

def check_symmetric(a: np.ndarray) -> bool:
    """Check if a matrix is symmetric.

    Parameters
    ----------
    a : np.ndarray
        A matrix to check.

    Returns
    ----------
    bool: A matrix is symmetric.
    """
    if not _check_matrix(a):
        return False

    if not _check_matrix_is_square(a):
        return False

    if not _check_square_matrix_is_symmetric(a):
        return False

    return True

def _check_binary(a: np.ndarray) -> bool:
    """Check if np.ndarray is binary.

    Parameters
    ----------
    a: np.ndarray

    Returns
    ----------
    bool: np.ndarray is binary.
    """
    return ((a == 0) | (a == 1)).all()

def check_adjacency_matrix(a: np.ndarray) -> bool:
    """Check if a matrix is adjacency_matrix.

    Parameters
    ----------
    a : np.ndarray
        A matrix to check.

    Returns
    ----------
    bool: A matrix is adjacency_matrix.
    """
    if not check_symmetric(a):
        return False

    if not _check_binary(a):
        return False

    # nonzero diagonal - graph with loops
    if np.any(np.diag(a)):
        return False

    return True


def _pairwise_distances(
    X: np.ndarray,
    metric: Union[str, Callable] = "euclidean",
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Compute the pairwise distance matrix from a matrix X.

    Parameters
    ----------
    X : np.ndarray
        A matrix.

    metric : Union[str, Callable], optional, defaults = "euclidean"
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter,
        or a metric listed in sklearn pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    
    n_jobs : Optional[int], optional, defaults = None
        The number of jobs to use for the computation

    Returns
    ----------
    distances: np.ndarray
        The pairwise distance matrix.
    """
    assert _check_matrix(X)

    distances = pairwise_distances(X=X, metric=metric, n_jobs=n_jobs)

    return distances

def distances_to_adjacency_matrix(
    distances: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert a pairwise distance matrix to adjacency_matrix given threshold.

    Parameters
    ----------
    distances : np.ndarray
        A pairwise distance matrix.
    
    threshold : float
        Threshold to make graph edges.

    Returns
    ----------
    adjacency_matrix : np.ndarray
        The adjacency_matrix.
    """
    assert check_symmetric(distances)

    N = distances.shape[0]

    adjacency_matrix = (distances < threshold).astype(int) - np.eye(N, dtype=int)

    return adjacency_matrix

def span_tree_top_n_weights_idx(
    span_tree: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get indices of top n weights in the span tree.

    Parameters
    ----------
    span_tree : np.ndarray
        The span tree.

    n : int
        Top n weights to find.

    Returns
    ----------
    unravel_idx: Tuple[np.ndarray]
        Indices of top n weights in the span tree.
    """
    span_tree_shape = span_tree.shape
    N = span_tree_shape[0]

    if n == 0:
        array_1_to_N = np.array(range(N))
        unravel_idx = (array_1_to_N, array_1_to_N)

    else:
        ravel_span_tree_top_n_weights_idx = np.argpartition(
            a=span_tree.ravel(),
            kth=-n,
        )[-n:]

        unravel_idx = np.unravel_index(
            indices=ravel_span_tree_top_n_weights_idx,
            shape=span_tree_shape,
        )

    return unravel_idx

class ConnectedComponentsClustering(ClusterMixin, BaseEstimator):
    """
    Clustering with graph connected components.

    Parameters
    ----------
    threshold : float, default = 0
        Threshold to make graph edges.

    metric : Union[str, Callable], optional, defaults = "euclidean"
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter,
        or a metric listed in sklearn pairwise.PAIRWISE_DISTANCE_FUNCTIONS.

    n_jobs : Optional[int], optional, defaults = None
        The number of jobs to use for the computation. Defaults to None.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import ConnectedComponentsClustering
    >>> X = np.array([[0, 1], [1, 0], [1, 1]])
    >>> clustering = ConnectedComponentsClustering(threshold=0.275, metric="euclidean", n_jobs=-1).fit(X)
    >>> clustering
    ConnectedComponentsClustering(threshold=0.275, metric="euclidean", n_jobs=-1)
    """

    def __init__(
        self,
        threshold = 0,
        metric = "euclidean",
        n_jobs = None,
    ):
        self.threshold = threshold
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit graph-based clustering model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)

        y : Ignored
            Not used, present here for API consistency by convention.
        """

        X = self._validate_data(X, accept_sparse="csr")

        distances = _pairwise_distances(
            X=X,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

        adjacency_matrix = distances_to_adjacency_matrix(
            distances=distances,
            threshold=self.threshold,
        )

        graph = csr_matrix(adjacency_matrix)

        _, labels = connected_components(
            csgraph=graph,
            directed=True,
            return_labels=True,
        )

        self.labels_ = labels

        return self

    def fit_predict(self, X, y=None):
        """Fit graph-based clustering model and return labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y)
        return self.labels_
    
class SpanTreeConnectedComponentsClustering(ClusterMixin, BaseEstimator):
    """
    Clustering with graph span tree connected components.

    Parameters
    ----------
    n_clusters : int, defaults = 3
        The number of clusters to find. 

    metric : Union[str, Callable], optional, defaults = "euclidean"
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter,
        or a metric listed in sklearn pairwise.PAIRWISE_DISTANCE_FUNCTIONS.

    n_jobs : Optional[int], optional, defaults = None
        The number of jobs to use for the computation. Defaults to None.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import SpanTreeConnectedComponentsClustering
    >>> X = np.array([[0, 1], [1, 0], [1, 1]])
    >>> clustering = SpanTreeConnectedComponentsClustering(n_clusters=3, metric="euclidean", n_jobs=-1).fit(X)
    >>> clustering
    SpanTreeConnectedComponentsClustering(n_clusters=3, metric="euclidean", n_jobs=-1)
    """
    
    def __init__(
        self,
        n_clusters=3,
        metric="euclidean",
        n_jobs= None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.n_jobs = n_jobs
    

    def fit(self, X, y=None):
        """Fit graph-based clustering model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or \
                (n_samples, n_samples)

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        X = self._validate_data(X, accept_sparse="csr")
        return self._fit(X)

    def _fit(self, X):
        """Fit without validation

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError(
                "n_clusters should be an integer greater than 0. %s was provided."
                % str(self.n_clusters)
            )

        if self.n_clusters is None:
            raise ValueError(
                "n_clusters should not be None for SpanTreeConnectedComponentsClustering."
            )
        
        distances = _pairwise_distances(
            X=X,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

        span_tree = minimum_spanning_tree(distances).toarray()

        top_n_weights_idx = span_tree_top_n_weights_idx(
            span_tree=span_tree,
            n=self.n_clusters - 1,
        )

        graph_n_clusters = span_tree.copy()

        graph_n_clusters[top_n_weights_idx] = 0
        graph_n_clusters[graph_n_clusters > 0] = 1
        graph_n_clusters = graph_n_clusters.astype(int)

        graph = csr_matrix(graph_n_clusters)

        _, labels = connected_components(
            csgraph=graph,
            directed=False,
            return_labels=True,
        )

        self.labels_ = labels

        return self
    
    def fit_predict(self, X, y=None):
        """Fit graph-based clustering model and return labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y)
        return self.labels_