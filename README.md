# MatOCLUST

`MatOCLUST` is a Julia package for clustering matrix-valued data with outliers. It is the matrix-variate version of oclust. 

## Features

- Model-based clustering of matrix-valued observations
- Outlier detection and removal

## Installation

Install directly from GitHub:

```julia
using Pkg

Pkg.add(url="https://github.com/clarkkm2/MatOCLUST.jl")
```

## Loading the Package

```julia
using MatOCLUST
```

## Data Format

The package expects a collection of matrices stored as a vector:

```julia
using Random
using StatsBase
using Distributions
using LinearAlgebra
using MatOCLUST

# Number of observations
N = 200

# Matrix dimensions
r = 2
c = 4

# Number of clusters
G = 2

# Cluster means
M1 = [-2.6 -1.1 -0.5 -0.2;
       1.3  0.6  0.3  0.1]

M2 = [ 1.5  1.7  1.9  2.2;
      -3.7 -2.7 -2.0 -1.5]

# Row covariance matrices
U1 = Symmetric([2.0 0.0;
                0.0 1.0])

U2 = Symmetric([1.7 0.5;
                0.5 1.3])

# Column covariance matrix
V = Symmetric([
    1.00 0.50 0.25 0.13;
    0.50 1.00 0.50 0.25;
    0.25 0.50 1.00 0.50;
    0.13 0.25 0.50 1.00
])

# Generate cluster memberships
true_class = sample(1:G, Weights([0.5, 0.5]), N)

# Generate matrix-valued observations
X = [
    rand(MatrixNormal(
        true_class[i] == 1 ? M1 : M2,
        true_class[i] == 1 ? U1 : U2,
        V
    ))
    for i in 1:N
]


```

Each element of `X` corresponds to one observation.

## Basic Example

```julia
using matOCLUST

# Fit a two-component model with 10 candidate outliers
fit = matOCLUST(X, 2, 10)

# Estimated classifications
fit["classes"]

# K-L Divergence
fit["KL"]

# Compare estimated and true classes
crosstab(true_class, fit["classes"])
```

where

- `X` is a vector of matrices
- `3` is the number of clusters
- `10` is the maximum number of candidate outliers

## Main Functions

### `matOCLUST`

Run matrix outlier clustering.

```julia
matOCLUST(
    X,
    G,
    F;
    maxiter = 1000,
    eps = 1e-5,
    zinit = "kmeans",
    nsuccess = 10,
    parallel = true,
    prnt = true
)
```

#### Arguments

- `X`: vector of matrices
- `G`: number of clusters
- `F`: maximum number of outliers
- `maxiter`: maximum number of EM iterations
- `eps`: convergence criterion
- `zinit`: initialization option
- `nsuccess`: number of starts to try for each iteration of matOCLUST
- `parallel`: a Boolean specifying if subset log-likelihoods should be calculated in parallel
- `prnt`: a Boolean specifying if a progress bar should be displayed

### `miniEM`

Run multiple initializations of the EM algorithm. 

```julia
miniEM(...)
```

### `EM`

Runs the full EM algorithm.

```julia
EM(...)
```

### `crosstab`

Produces a contingency table comparing two classification vectors.

```julia
crosstab(predicted, truth)
```

## Output

The fitted model object contains information such as:

```julia
fit["classes"]
fit["bestmod"]
fit["numO"]
fit["outs"]
fit["allCand"]
fit["KL"]
```

## References

This package is the Julia implementation of:

> Clark, K. M., & McNicholas, P. D. (2024). *Clustering Three-Way Data with Outliers*. arXiv:2310.05288.

## License

GPL-2.0