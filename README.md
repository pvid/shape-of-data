# Shape of Data

### Very much work in progress

The purpose of this repository is to look under the hood of algorithms
that explore the shape of data.

The algorithms are not meant to be production ready. However, I am interested
in performance as well. I will be comparing the run-time of my Python
implementation with the C++ Dionysus library.

## Topics:
* Topological data analysis - more precisely persistent homology
* (TO DO) Laplacian eigenmaps (spectral embedding)
* Spectral clustering

## Why?

I wanted to play a bit with persistent homology, its implementation details
and relations to other method. However, due to the coputational complexity,
most of the libraries are written in C++ (like Dionysus, GUDHI, PHAT, etc.)
at best with a not very well documented  Python biding.

I wanted to tinker with the algorithms a bit and see how can
information from persistent homology be of use when finding and/or
diagnosing manifold embedding and clustering.

## Persistent homology

In a sentence, it measures geometrical complexity of a dataset using algebraic invariants
that have something to do with connectedness, loops, cavities and their high dimensional
counterparts. For a quick introduction with pictures, see [An introduction to persistent homology](http://bastian.rieck.ru/research/an_introduction_to_persistent_homology.pdf)

What am I implementing:
- Construction of a Vietoris-Rips complex (see paper [[1]](#references))
- Calculating the persistent homology over ![equation](http://latex.codecogs.com/gif.latex?\mathbb{Z}_2)

## Laplacian eigenmaps

I am following [[2]](#references). It uses a different algorithm
than the scikit-learn implementation of spectral embedding.

One version uses adjecency graph constructed by looking at ![equation](http://latex.codecogs.com/gif.latex?\epsilon)-neighborhoods,
which is the first step of construction of Vietoris-Rips complex.

Goals:
* Implement the algorithm and deal with unconnected adjecency graphs
  if possible (scikit-learn does not do that)
* How can pesistent homology inform the choice of ![equation](http://latex.codecogs.com/gif.latex?\epsilon) in the construction?
* How does do the eigenmaps behave with manifolds with non-zero Betti numbers?




## Spectral clustering

...

## References

I do plan to clean this up at some point. In the meantime:

[1] Zomorodian, A. (2010). [Fast construction of the Vietoris-Rips complex](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.210.426&rep=rep1&type=pdf).
 Computers & Graphics, 34(3), 263–271. https://doi.org/10.1016/j.cag.2010.03.007

[2] Belkin, M., & Niyogi, P. (2003). [Laplacian Eigenmaps for Dimensionality Reduction and Data Representation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8814&rep=rep1&type=pdf).
Neural Computation, 15(6), 1373–1396. https://doi.org/10.1162/089976603321780317
