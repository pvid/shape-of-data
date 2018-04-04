"""Submodule containing the PointCloud class."""

import numpy as np

from .rips import pairwise_dist, neighbor_graph, generate_filtration


class PointCloud(object):
    """
    Wrapper for raw data.

    Holds raw data, intermediate calculations and provides an interface various methods
    that can be applied to the date.
    It tries to reuse calculations, mainly filtration generation due to its cost.

    """

    def __init__(self, data, distance_only=False):
        """
        Initialize point cloud with raw data or a distance matrix.

        Parameters
        ----------
        data: ndarray
            Matrix whose rows are datapoints or a square distance matrix.

        distance_only: boolean
            Specifies if raw data or a distance matrix was passed.

        """
        self._datapoints = None
        self._dist = None
        self._eps = None
        self._neighborhood_graph = None
        self._filtration_dim = None
        self._filtration = None

        if distance_only:
            shape = data.shape
            if shape[0] != shape[1]:
                raise ValueError('The distance matrix is nor square.')
            if np.count_nonzero((np.diagonal(data))) > 0:
                raise ValueError('The dianonal is not all zeros.')
            self._dist = data
            self._datapoints = None
        else:
            self._datapoints = data

    def distance_matrix(self):
        """
        Return the distance matrix of the point cloud.

        Returns
        -------
        dist: ndarray
            Square matrix of pairwise euclidean distances.

        """
        if self._dist is None:
            self._dist = pairwise_dist(self._datapoints)
        return self._dist

    def neighborhood_graph(self, eps):
        """
        Construct neighborhood graph of the point cloud.

        Parameters
        ----------
        eps: float
            The distance specifying, when points are close enough
            to be neighbors.

        Returns
        -------
        graph: dict of sets of int
            Dictionary containing adjacency sets of a vertex.

        """
        if self._eps is None:
            self._eps = eps
            self._neighborhood_graph = neighbor_graph(self.distance_matrix(), eps)
        if eps != self._eps:
            self._neighborhood_graph = neighbor_graph(self.distance_matrix(), eps)
        return self._neighborhood_graph

    def filtration(self, eps=None, dim=None):
        """
        Build filtration of a Vietoris-Rips complex.

        The algorithm uses backtracking from the Bron-Kerborsch algorithm.

        Parameters
        ----------
        eps: float
            The distance specifying, when points are close enough
            to be neighbors.
            If None and a neighborhood graph has been built, it is inferred
            as the epsilon used there.


        dim: int
            Dimension of the resulting complex. If you want to explore topological
            features of dimension k, Build a filtation of dimension k+1.
            If None, the dimension is inferred as number of coordinates
            of datapoints + 2.

        Returns
        -------
        filtration: List of tuples
            First element of the tuple represents a face by a tuple
            of vertices.
            Second represents the weight of the face.
            Sorted by second element.

        """
        # Try to infer values of eps and dim if needed
        if eps is None:
            if self._eps is None:
                raise ValueError('Cannot infer an epsilon value.')
            eps = self._eps

        if dim is None:
            if self._datapoints is None:
                raise ValueError('Cannot infer dimension.')
            dim = self._datapoints.shape[1] + 1

        # Reuse calculated filtration if possible
        if self._filtration:
            if eps == self._eps and self._filtration_dim == dim:
                return self._filtration
            if eps < self._eps and self._filtration_dim < dim:
                filt_eps = filter(lambda x: x[1] < eps, self._filtration)
                filt = filter(lambda x: len(x[0]) < dim+2, filt_eps)
                return list(filt)

        self._filtration_dim = dim
        self._filtration = (
            generate_filtration(
                self.neighborhood_graph(eps),
                self.distance_matrix(),
                dim)
        )
        return self._filtration
