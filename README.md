# Kempe-Swap K-Means for Semi-Supervised Clusterwise Regression (KSKM-CLR)

A scalable, near-optimal algorithm for **semi-supervised clusterwise linear regression** under hard **must-link (ML)** and **cannot-link (CL)** constraints.

> Based on the papers:
> - **"Kempe Swap K-Means: A Scalable Near-Optimal Solution for Semi-Supervised Clustering"**  
>   Yuxuan Ren, Shijie Deng — Georgia Institute of Technology  
>   [arXiv:2603.27417](https://arxiv.org/abs/2603.27417)
> - **"Application of Semi-supervised Learning in Energy Markets"**  
>   Yuxuan Ren — Georgia Institute of Technology

---

## Overview

**Clusterwise regression** partitions data into k groups and fits a separate linear regression model within each group, minimising the total sum of squared regression residuals (SSR).  When domain knowledge constrains which data points may or may not share the same regression regime, this becomes **semi-supervised clusterwise regression (SSCLR)**:

- **Must-link (ML):** a set of data points must belong to the same cluster (and thus share the same regression model).
- **Cannot-link (CL):** two super-nodes (ML equivalence classes) must be placed in different clusters.

### Relationship to Clustering

Standard K-Means minimises within-cluster sum of squares (WCSS) using Euclidean centroids. KSKM-CLR instead minimises total SSR by fitting OLS regression coefficients B_j as the "centroid" of each cluster j.  The assignment distance from super-node v to cluster j is the regression SSE of all data points in v under B_j.

### Constraint Representation via Super-Nodes

Must-link constraints are transitive: if i and j must be linked, and j and h must be linked, then i, j, and h all belong to the same super-node.  The algorithm therefore takes constraints already pre-processed into super-nodes:

- **`ml_supernodes`** — a list of super-nodes. `ml_supernodes[i]` is the list of 0-indexed data-point indices forming super-node i. Single-point lists represent unconstrained data points.
- **`cl_supernodes`** — a list of cannot-link super-node pairs. Each `(i, j)` entry means super-node `i` and super-node `j` must be in different clusters. Indices refer to positions in `ml_supernodes`.

Cannot-link constraints between super-nodes reduce the clustering problem to a **graph k-colouring** problem on the cannot-link graph G = (V, E).

---

## Algorithm

### Core Idea

KSKM-CLR adopts a K-Means-style iterative framework:

1. **Assignment step (KSAssignment):** Identifies all improving **Kempe chains** between cluster pairs and selects a compatible subset by solving a **Maximum Weight Independent Set (MWIS)** problem.  This guarantees that each swap preserves the valid k-colouring (feasibility is never violated) while reducing total SSR.

2. **Centroid update step (CentroidUpdate):** Recomputes per-cluster OLS regression coefficients B_j from the current assignment and rebuilds the SSE distance matrix D[v, j] using batched GPU einsum.

### Escape from Local Optima

Two controlled centroid mutations help avoid poor local optima:

- **Centroid perturbation (KempeChainMutation_distr):** Adds scaled noise drawn from the null space of X_j to B_j.  Clusters with larger residuals receive stronger perturbations, encouraging exploration where the regression fit is weakest.
- **Centroid reposition (CentroidUpdate with `reposition=True`):** Replaces the best-fitting cluster's distance column with the worst-fitting cluster's column, forcing a larger jump in the solution neighbourhood.

### Initialization

Super-node assignments are initialised using the **DSATUR heuristic** (Algorithm 5 in the paper): super-nodes are processed in non-increasing order of saturation degree and each is assigned to its nearest feasible centroid.

---

## Installation

### Requirements

```
numpy
scipy
numba
cupy        # GPU-accelerated batch einsum for distance matrix computation
gurobipy    # requires a valid Gurobi license
```

Install dependencies:

```bash
pip install numpy scipy numba gurobipy
# install cupy for your CUDA version, e.g.:
pip install cupy-cuda12x
```

> **Note:** Gurobi requires a license. Academic licenses are available free of charge at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Usage

```python
import numpy as np
from KSKM_CLR_SuperNodes import KSKM

# Feature matrix (n samples, q features) and response matrix (n samples, p responses)
X = np.array([...])  # shape (n, q)
Y = np.array([...])  # shape (n, p)

# Must-link super-nodes: each entry is a list of 0-indexed data-point indices.
# Points in the same super-node are forced into the same cluster.
# Example: super-node 0 contains points {0, 3, 7}; super-node 1 contains point {1}, etc.
ml_supernodes = [
    [0, 3, 7],   # super-node 0 — points 0, 3, 7 must be in the same cluster
    [1],         # super-node 1 — unconstrained singleton
    [2, 5],      # super-node 2 — points 2 and 5 must be in the same cluster
    [4],         # super-node 3 — unconstrained singleton
    [6],         # super-node 4 — unconstrained singleton
]

# Cannot-link super-node pairs: each entry (i, j) means super-node i and super-node j
# must be assigned to different clusters. Indices refer to ml_supernodes positions.
cl_supernodes = [
    (0, 2),  # super-node 0 and super-node 2 must be in different clusters
    (1, 3),  # super-node 1 and super-node 3 must be in different clusters
]

# Run KSKM-CLR
membership_final, membership = KSKM(
    random_state=42,
    X=X,
    Y=Y,
    ml_supernodes=ml_supernodes,
    cl_supernodes=cl_supernodes,
    steps_mutation=200,      # maximum number of mutation iterations
    k=4,                     # number of clusters / regression components
    steps_back_to_best=5,    # revert to best after this many non-improving mutations
    steps_no_improvement=10, # terminate after this many total non-improving mutations
    verbose=False,
    time_limit=3600,         # seconds
    reposition_frequency=5,  # reposition every 5th mutation; otherwise perturb
    random_centroid_scale=10,# noise scale for centroid perturbation
    weight_supernodes=False, # set True to equalise super-node contributions by size
)
# membership_final: cluster label for each data point (shape n,)
# membership:       cluster label for each super-node (can be reused as warm start)
```

See [`example.py`](example.py) for a complete runnable example loading super-node data from disk.

---

## API Reference

### `KSKM(random_state, X, Y, ml_supernodes, cl_supernodes, steps_mutation, k, ...)`

Main entry point.  Runs KSKM-CLR: Kempe-swap local search with centroid mutations, minimising total OLS regression SSR subject to ML/CL constraints.

| Parameter | Type | Description |
|---|---|---|
| `random_state` | `int` | NumPy random seed |
| `X` | `ndarray (n, q)` | Feature matrix (intercept column prepended internally) |
| `Y` | `ndarray (n, p)` | Response matrix |
| `ml_supernodes` | `list[list[int]]` | Must-link super-nodes; `ml_supernodes[i]` = data-point indices in super-node i |
| `cl_supernodes` | `list[tuple[int,int]]` | Cannot-link pairs of super-node indices |
| `steps_mutation` | `int` | Maximum mutation iterations |
| `k` | `int` | Number of clusters |
| `steps_back_to_best` | `int` | Revert to best solution after this many non-improving mutations |
| `steps_no_improvement` | `int` | Terminate after this many non-improving mutations |
| `verbose` | `bool` | Print SSR during descent |
| `time_limit` | `float` | Wall-clock time limit (seconds) |
| `membership` | `ndarray \| None` | Warm-start super-node assignments; `None` triggers DSATUR init |
| `reposition_frequency` | `int` | Apply reposition every N-th mutation; else perturb |
| `random_centroid_scale` | `float` | Noise scale for centroid perturbation |
| `weight_supernodes` | `bool` | Weight points by 1/|super-node| to equalise super-node influence |

**Returns:**
- `membership_final` — `ndarray (n,)`: cluster label for each data point (0-indexed), or `[]` if no feasible initialisation was found.
- `membership` — `ndarray (n_supernodes,)`: cluster label per super-node (reusable as warm start).

---

## Key Internal Components

| Function | Role |
|---|---|
| `preprocessing` | Converts `ml_supernodes` / `cl_supernodes` into CSR arrays and the cannot-link adjacency list |
| `sub_adj_classification` | Decomposes the cannot-link graph into singletons, cliques, and general components |
| `DSATUR` | DSATUR graph-colouring heuristic for feasible initialisation |
| `CentroidUpdate` | Fits per-cluster OLS regression centroids; builds GPU-accelerated SSE distance matrix D; handles empty clusters, reposition, and perturbation |
| `KSAssignment` | One round of Kempe-swap assignment: enumerates improving swaps via `build_MWSP_core` and selects compatible subset via Gurobi MWIS |
| `build_MWSP_core` | Numba-JIT function that enumerates Kempe chains and their conflict graph for one connected component |
| `KempeChainMWSP` | Full descent loop: alternates CentroidUpdate + KSAssignment until local SSR optimum |
| `KempeChainMutation_target_centroids` | Runs KSAssignment to convergence under a fixed D (used after mutations) |
| `KempeChainMutation_distr` | Centroid-perturbation mutation followed by descent |
| `recover_ml_from_membership` | Expands super-node assignments back to individual data-point labels |

---

## Application: LMP Forecasting in Energy Markets

The primary application of KSKM-CLR is probabilistic **Locational Marginal Price (LMP) forecasting** in electricity markets.  From the DC-OPF multiparametric programming perspective, the load input space partitions into **critical regions**, within each of which LMP is an affine function of system loads.  KSKM-CLR recovers aggregated critical regions directly from historical data by:

- **Must-linking** operating points that share identical binding transmission constraint sets (same system pattern).
- **Cannot-linking** super-nodes whose active constraint sets differ by more than $\Delta$ constraint (differing regimes).
- Fitting a separate affine LMP model per cluster to minimise total regression SSR.

---

## Performance

KSKM-CLR achieves near-optimal SSR partitions while remaining orders of magnitude faster than exact ILP-based methods (PCCC, BLPKMCC) for large-scale datasets.  The MWIS subproblem solved per iteration is far cheaper than the full fixed-centroid graph-colouring ILP used by competing methods, enabling KSKM-CLR to scale to datasets with tens of thousands of data points.

---

## Citation

```bibtex
@article{ren2026kskm,
  title   = {Kempe Swap K-Means: A Scalable Near-Optimal Solution for Semi-Supervised Clustering},
  author  = {Ren, Yuxuan and Deng, Shijie},
  journal = {arXiv preprint arXiv:2603.27417},
  year    = {2026}
}
```
