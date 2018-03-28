# Shape of Data

### Very much work in progress

The purpose of this repository is to look under the hood of algorithms
that explore the shape of data.

## Topics:
* Topological data analysis - more precisely persistent homology
* (TO DO) Laplacian eigenmaps (spectral embedding), Locally linear embedding
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
- Construction of a Vietoris-Rips complex (see paper [[FAST]](#references))
- Calculating the persistent homology over <img src="https://latex.codecogs.com/svg.latex?\mathbb{Z}_2" title="Z2" />

## Laplacian eigenmaps
...

See [EIGEN](#references)

## Spectral clustering

...

## References

I do plan to clean this up at some point. In the meantime:

[FAST] Zomordian: [Fast Construction of the Vietoris-Rips Complex](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.210.426&rep=rep1&type=pdf)

[EIGEN] Belkin, Niyogi [Laplacian Eigenmaps for Dimensionality Reduction and Data
Representation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.192.8814&rep=rep1&type=pdf)
