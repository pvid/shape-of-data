"""Submodule for calculating the Vietoris-Rips filtration of a point cloud."""

from itertools import combinations
from operator import itemgetter

import numpy as np


def rips_complex(dim, eps, data=None, dist=None,):
    """
    Generate Vietoris-Rips.

    The function uses either raw point coordinates ('data' argument)
    or distance matrix ('dist' argument),

    Parameters
    ----------
    data: ndarray
        Numpy array, whose rows are point coordinates

    dist: ndarray
        Matrix of pairwise distances

    dim: int
        Dimension of the resulting complex

    eps: float
        The distance specifying, when points are close enough
        to be neighbors.

    Returns
    -------
    filtration: List of tuples
        First element of the tuple represents a face by a tuple
        of vertices.
        Second represents the weight of the face.
        Sorted by second element.

    """
    if dist is None and data is None:
        raise ValueError("Either 'data', or 'dist' is needed.")

    if dist is None:
        dist = _pairwise_dist(data)

    graph = _neighbor_graph(dist, eps)
    filtration = _generate_filtration(graph, dist, dim)

    return filtration


def _pairwise_dist(X):
    """
    Calculate paiwise euclidean distances.

    Parameters
    ----------
    X: ndarray
        Numpy array, whose rows are point coordinates

    Returns
    -------
    dist: ndarray
        Square matrix of pairwise euclidean distances.

    """
    XX = np.einsum('ij,ij->i', X, X)[:, None]

    # Ensure positive entries
    dist = np.maximum(XX + XX.T - 2*X.dot(X.T), 0)

    # Ensure 0 on diagonal
    np.fill_diagonal(dist, 0)
    return np.sqrt(dist)


def _neighbor_graph(dmat, eps):
    """
    Build the neighborhood graph.

    Parameters
    ----------
    dmat: ndarray
        A square matrix of pairwise distances of points.

    eps: float
        The distance specifying, when points are close enough
        to be neighbors.

    Returns
    -------
    graph: dict of sets of int
        Dictionary containing adjacency sets of a vertex.

    """
    graph = dict()
    n = dmat.shape[0]
    for i in range(n):
        graph[i] = set()
    for i in range(n):
        for j in range(i+1, n):
            if dmat[i, j] < eps:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _generate_filtration(graph, dist, dim):
    """
    Build filtration of a Vietoris-Rips complex.

    The algorithm uses backtracking from the Bron-Kerborsch algorithm.

    Parameters
    ----------
    graph: dict of sets of int
        Dictionary containing adjacency sets of a vertex.

    dist: ndarray
        A square matrix of pairwise distances of points.

    dim: int
        Dimension of the resulting complex

    Returns
    -------
    filtration: List of tuples
        First element of the tuple represents a face by a tuple
        of vertices.
        Second represents the weight of the face.
        Sorted by second element.

    """
    simplices = []

    def bron_kerborsch(current, candidate, discard):

        if len(current) == dim+1:
            simplices.append(tuple(sorted(current)))
            return

        if candidate or discard:
            for vertex in candidate.copy():
                bron_kerborsch(current.union(set([vertex])),
                               candidate.intersection(graph[vertex]),
                               discard.intersection(graph[vertex]))
                candidate.remove(vertex)
                discard.add(vertex)

        simplices.append(tuple(sorted(current)))

    bron_kerborsch(set(), set(graph.keys()), set())

    simplices = sorted(simplices, key=lambda x: len(x))[1:]

    val_memo = dict()
    filtration = []

    for simplex in simplices:
        k = len(simplex)

        if k == 1:
            filtration.append((simplex, 0))
            continue

        if k == 2:
            i, j = simplex
            val = dist[i, j]
            val_memo[simplex] = val
            filtration.append((simplex, val))
            continue

        val = max(val_memo[face] for face in combinations(simplex, k-1))
        val_memo[simplex] = val
        filtration.append((simplex, val))

    return sorted(filtration, key=itemgetter(1))


def persistent_homology(filtration, dim):
    """Compute persistent homology

    Parameters
    ----------
    filtration: List of tuples
        First element of the tuple represents a face by a tuple
        of vertices.
        Second represents the weight of the face.
        Sorted by second element.

    dim: int
        Dimension of the resulting complex


    Returns
    -------
    intervals:list of lists of tuples
        Elements are lists with intervals representing homological features

    """
    simp_idx = {s[0]: i for i, s in enumerate(filtration)}

    intervals = [[] for i in range(dim+1)]

    T = dict()
    marked = set()

    inf = float('inf')

    def remove_pivot_rows(simplex):
        k = len(simplex) - 1
        if k == 0:
            return set()
        d = set(face for face in combinations(simplex, k))
        d = d.intersection(marked)

        while len(d) > 0:
            i = max(simp_idx[face] for face in d)
            if i not in T.keys():
                break
            d = d.symmetric_difference(T[i][1])
        return d

    for j, (simplex, _) in enumerate(filtration):
        d = remove_pivot_rows(simplex)
        if len(d) == 0:
            marked.add(simplex)
        else:
            i = max(simp_idx[face] for face in d)
            k = len(filtration[i][0]) - 1
            T[i] = (j, d)
            a = filtration[i][1]
            b = filtration[j][1]
            if a != b:
                intervals[k].append((a, b))

    for simpl in marked:
        j = simp_idx[simpl]
        if j not in T.keys():
            k = len(filtration[j][0])-1
            intervals[k].append((filtration[j][1], inf))

    return intervals
