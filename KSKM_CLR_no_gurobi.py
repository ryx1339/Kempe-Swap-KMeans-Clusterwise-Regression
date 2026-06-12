from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import copy
from numba import njit
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import cupy as cp

@njit(cache=True, fastmath=True)
def build_MWSP_core(D, neighbors, neighbors_idx, membership_vertices, k, n_vertices, n_swaps_cap):
    eps = 1e-6
    len_colors = np.zeros(k, dtype=np.uint32)
    color_class_arr = np.empty((k, n_vertices), dtype=np.uint32)
    for v in range(n_vertices):
        c = membership_vertices[v]
        color_class_arr[c][len_colors[c]] = v
        len_colors[c] += 1
    s_ids_arr = np.empty((k, n_swaps_cap), dtype=np.uint32)
    s_ids_arr_idx = np.zeros(k, dtype=np.uint32)
    points_in_swaps_arr = np.empty((n_vertices, k-1), dtype=np.uint32)
    points_in_swaps_arr_idx = np.zeros(n_vertices, dtype=np.uint32)
    swaps_i_arr = np.empty(n_swaps_cap, np.uint32)
    swaps_j_arr = np.empty(n_swaps_cap, np.uint32) 
    swaps_Hi_arr = np.empty((k-1) * n_vertices, np.uint32) 
    swaps_Hj_arr = np.empty((k-1) * n_vertices, np.uint32)
    swaps_Hi_index_arr = np.empty(n_swaps_cap + 1, np.uint32)
    swaps_Hj_index_arr = np.empty(n_swaps_cap + 1, np.uint32)
    swaps_Hi_index_arr[0] = 0
    swaps_Hj_index_arr[0] = 0
    swaps_weights_arr = np.empty(n_swaps_cap, dtype=np.float64)
    len_adj_cap = int(n_swaps_cap * (n_swaps_cap-1) / 2)
    adj_swaps_u_arr = np.empty(len_adj_cap, np.uint32)
    adj_swaps_v_arr = np.empty(len_adj_cap, np.uint32)
    len_adj = 0
    visited = np.zeros(n_vertices, dtype=np.uint32)
    q_v = np.empty(n_vertices, dtype=np.uint32)
    q_side = np.empty(n_vertices, dtype=np.bool_)
    H_i_neighbor = np.full(n_vertices, -1, dtype=np.int64)
    H_j_neighbor = np.full(n_vertices, -1, dtype=np.int64)
    stamp = 0
    s_id = 0
    len_swaps_Hi = 0
    len_swaps_Hj = 0
    for i in range(k):
        len_i = len_colors[i]
        for j in range(i + 1, k):
            len_j = len_colors[j]
            if len_i == 0:
                for v_idx in range(len_j):
                    v = color_class_arr[j, v_idx]
                    delta = -(D[v, j] - D[v, i])
                    if delta < -eps:
                        swaps_i_arr[s_id] = i
                        swaps_j_arr[s_id] = j
                        swaps_Hj_arr[len_swaps_Hj] = v
                        len_swaps_Hj += 1
                        swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                        swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                        swaps_weights_arr[s_id] = delta
                        neighbor_from = neighbors_idx[v]
                        neighbor_to = neighbors_idx[v+1]
                        points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                        points_in_swaps_arr_idx[v] += 1
                        if neighbor_to > neighbor_from:
                            for t in range(neighbor_from,neighbor_to):
                                H_j_neighbor[neighbors[t]] = s_id
                            for s in s_ids_arr[i, :s_ids_arr_idx[i]]:
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if (ii == i) and (j != jj):
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif jj == i:
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[i, s_ids_arr_idx[i]] = s_id
                            s_ids_arr_idx[i] += 1
                        s_id += 1
                        if s_id >= n_swaps_cap:
                            return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
            elif len_j == 0:
                for v_idx in range(len_i):
                    v = color_class_arr[i,v_idx]
                    delta = (D[v, j] - D[v, i])
                    if delta < -eps:
                        swaps_i_arr[s_id] = i
                        swaps_j_arr[s_id] = j
                        swaps_Hi_arr[len_swaps_Hi] = v
                        len_swaps_Hi += 1
                        swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                        swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                        swaps_weights_arr[s_id] = delta
                        neighbor_from = neighbors_idx[v]
                        neighbor_to = neighbors_idx[v+1]
                        points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                        points_in_swaps_arr_idx[v] += 1
                        if neighbor_to > neighbor_from: 
                            for t in range(neighbor_from, neighbor_to):
                                H_i_neighbor[neighbors[t]] = s_id
                            for s in s_ids_arr[j, :s_ids_arr_idx[j]]: # swaps involving j
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if ii == j:
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif (jj == j) and (i != ii):
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[j, s_ids_arr_idx[j]] = s_id
                            s_ids_arr_idx[j] += 1
                        s_id += 1
                        if s_id >= n_swaps_cap:
                            return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
            else:
                stamp += 1
                for v_idx in range(len_i):
                    start = color_class_arr[i,v_idx]
                    if visited[start] == stamp:
                        continue
                    head = 0
                    tail = 0
                    q_v[tail] = start
                    q_side[tail] = True
                    tail += 1
                    visited[start] = stamp
                    delta = 0.0
                    while head < tail:
                        v = q_v[head]
                        side = q_side[head]
                        head += 1
                        if side:
                            delta += (D[v, j] - D[v, i])
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == j) and (visited[u] != stamp):
                                    visited[u] = stamp
                                    q_v[tail] = u
                                    q_side[tail] = False
                                    tail += 1
                        else:
                            delta -= (D[v, j] - D[v, i])
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == i) and (visited[u] != stamp):
                                    visited[u] = stamp
                                    q_v[tail] = u
                                    q_side[tail] = True
                                    tail += 1
                    if delta < -eps:
                        swaps_i_arr[s_id] = i
                        swaps_j_arr[s_id] = j
                        swaps_weights_arr[s_id] = delta
                        H_i_neighbor_non_empty = False
                        H_j_neighbor_non_empty = False
                        for idx in range(tail):
                            v = q_v[idx]
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                            points_in_swaps_arr_idx[v] += 1
                            if q_side[idx]:
                                swaps_Hi_arr[len_swaps_Hi] = v
                                len_swaps_Hi += 1
                                if neighbor_to > neighbor_from:
                                    H_i_neighbor_non_empty = True
                                    for t in range(neighbor_from, neighbor_to):
                                        H_i_neighbor[neighbors[t]] = s_id
                            else:
                                swaps_Hj_arr[len_swaps_Hj] = v
                                len_swaps_Hj += 1
                                if neighbor_to > neighbor_from:
                                    H_j_neighbor_non_empty = True
                                    for t in range(neighbor_from, neighbor_to):
                                        H_j_neighbor[neighbors[t]] = s_id
                        swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                        swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                        if H_i_neighbor_non_empty: 
                            for s in s_ids_arr[j, :s_ids_arr_idx[j]]: # swaps involving j
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if ii == j:
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif (jj == j) and (i != ii):
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[j, s_ids_arr_idx[j]] = s_id
                            s_ids_arr_idx[j] += 1
                        if H_j_neighbor_non_empty:
                            for s in s_ids_arr[i, :s_ids_arr_idx[i]]:
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if (ii == i) and (j != jj):
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif jj == i:
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[i, s_ids_arr_idx[i]] = s_id
                            s_ids_arr_idx[i] += 1
                        s_id += 1
                        if s_id >= n_swaps_cap:
                            return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
                for v_idx in range(len_j):
                    v = color_class_arr[j,v_idx]
                    if visited[v] != stamp:
                        delta = -(D[v, j] - D[v, i])
                        if delta < -eps:
                            swaps_i_arr[s_id] = i
                            swaps_j_arr[s_id] = j
                            swaps_Hj_arr[len_swaps_Hj] = v
                            len_swaps_Hj += 1
                            swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                            swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                            swaps_weights_arr[s_id] = delta
                            points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                            points_in_swaps_arr_idx[v] += 1
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            if neighbor_to > neighbor_from:
                                for t in range(neighbor_from, neighbor_to):
                                    H_j_neighbor[neighbors[t]] = s_id
                                for s in s_ids_arr[i, :s_ids_arr_idx[i]]:
                                    ii = swaps_i_arr[s]
                                    jj = swaps_j_arr[s]
                                    if (ii == i) and (j != jj):
                                        jj_from = swaps_Hj_index_arr[s]
                                        jj_to = swaps_Hj_index_arr[s+1]
                                        for u_index in range(jj_from, jj_to):
                                            u = swaps_Hj_arr[u_index]
                                            if visited[u] == stamp:
                                                break
                                            if H_j_neighbor[u] == s_id:
                                                adj_swaps_u_arr[len_adj] = s
                                                adj_swaps_v_arr[len_adj] = s_id
                                                len_adj += 1
                                                break
                                    elif jj == i:
                                        ii_from = swaps_Hi_index_arr[s]
                                        ii_to = swaps_Hi_index_arr[s+1]
                                        for u_index in range(ii_from, ii_to):
                                            u = swaps_Hi_arr[u_index]
                                            if visited[u] == stamp:
                                                break
                                            if H_j_neighbor[u] == s_id:
                                                adj_swaps_u_arr[len_adj] = s
                                                adj_swaps_v_arr[len_adj] = s_id
                                                len_adj += 1
                                                break
                                s_ids_arr[i, s_ids_arr_idx[i]] = s_id
                                s_ids_arr_idx[i] += 1
                            s_id += 1
                            if s_id >= n_swaps_cap:
                                return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
    return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]

@njit(cache=True)
def dsatur_color_numba(neighbors, neighbors_idx, n):
    """
    Equivalent to the Python dsatur_color(adj):
    - unbounded colors (up to n-1)
    - tie-break: max saturation, then max degree, then max vertex id
    Returns: colors (int64[n]), always feasible for simple graphs.
    """

    colors = np.full(n, -1, np.int64)

    degree = np.empty(n, np.int64)
    for i in range(n):
        degree[i] = neighbors_idx[i+1] - neighbors_idx[i]

    # neigh_colors[v, c] = 1 if v has some colored neighbor with color c
    neigh_colors = np.zeros((n, n), np.bool_)

    # maintain saturation count incrementally (exactly = number of 1s in row)
    sat = np.zeros(n, np.int64)

    n_colored = 0
    while n_colored < n:
        # pick vertex with max (sat, degree, id)
        best_v = -1
        best_sat = -1
        best_deg = -1

        for u in range(n):
            if colors[u] != -1:
                continue
            su = sat[u]
            du = degree[u]
            if (su > best_sat or
                (su == best_sat and (du > best_deg or
                                     (du == best_deg and u > best_v)))):
                best_sat = su
                best_deg = du
                best_v = u

        v = best_v
        if v == -1:
            # shouldn't happen
            print('DSATUR error')
            return colors

        # choose smallest unused color (0,1,2,...)
        chosen = -1
        for c in range(n):
            if not neigh_colors[v, c]:
                chosen = c
                break

        colors[v] = chosen
        n_colored += 1

        # update neighbors' saturation sets: neigh_colors[w].add(chosen)
        neighbor_from = neighbors_idx[v]
        neighbor_to = neighbors_idx[v+1]
        for t in range(neighbor_from, neighbor_to):
            w = neighbors[t]
            if colors[w] == -1:
                if not neigh_colors[w, chosen]:
                    neigh_colors[w, chosen] = True
                    sat[w] += 1

    return colors


def InitAssign_Singletons_and_Cliques(membership, sub_adjs, D):
    verts = sub_adjs['singletons']
    D_verts = D[verts]
    best_j = np.argmin(D_verts, axis=1)
    for v, j_new in zip(verts, best_j):
        membership[v] = j_new

    for verts in sub_adjs['cliques']:
        D_verts = D[verts]
        _, best_j = linear_sum_assignment(D_verts)
        for v, j_new in zip(verts, best_j):
            membership[v] = j_new
    return membership

def Assign_Singletons_and_Cliques(membership, sub_adjs, D, n_s, eps = 1e-6):
    verts = sub_adjs['singletons']
    D_verts = D[verts]
    best_j = np.argmin(D_verts, axis=1) 
    current_j = membership[verts]
    # move_mask = (current_j != best_j)
    best_cost = D[verts, best_j[verts]]
    current_cost = D[verts, current_j]

    move_mask = (best_j[verts] != current_j) & (best_cost < current_cost - eps)

    moved_verts = verts[move_mask]
    if moved_verts.size:
        membership[moved_verts] = best_j[move_mask]
        n_s += 1

    for verts in sub_adjs['cliques']:
        D_verts = D[verts]
        current_j = membership[verts]
        _, best_j = linear_sum_assignment(D_verts)
        move_mask = (current_j != best_j)
        moved_verts = verts[move_mask]
        if moved_verts.size:
            membership[moved_verts] = best_j[move_mask]
            n_s += 1
    return membership, n_s

def CentroidUpdate(
    membership, ml_map, X, Y, k,
    ml_array, ml_array_idxs,
    reposition=False,
    rcond=1e-12,
    random_centroid_scale = 0
):
    """
    Returns:
      if not reposition: (D, total_ssr_rounded)
      if reposition: D (D, total_ssr_rounded), (with best column replaced by worst column)
    """
    def ols_fit_sse(Xj: np.ndarray, Yj: np.ndarray):
        """
        OLS fit for multi-output:
        B_hat: (q, p)
        sse_col: (p,)  column-wise SSE
        dof: scalar = n_j - rank(Xj)   (safe for rank deficiency)

        When n < q (underdetermined), perturbs the min-norm solution by a
        random draw from the null space.  Residuals/SSE are invariant to this
        because X @ V_null = 0 by definition.
        """
        n, q = Xj.shape
        Bj, res, rank, _ = np.linalg.lstsq(Xj, Yj, rcond=None)

        # res is non-empty only when n > q and rank == q (lstsq docs).
        # Residuals are invariant to null-space perturbation, so compute sse_col
        # from the unperturbed Bj before the perturbation branch.
        if res.size:
            sse_col = res
        else:
            R = Yj - Xj @ Bj
            sse_col = np.sum(R * R, axis=0)
            # sse_col = 0

        if rank < q:                          # null space exists
            # full_matrices=False: never allocates the (n×n) U matrix.
            _, _, Vt = np.linalg.svd(Xj, full_matrices=True)
            V_null = Vt[rank:].T              # (q, null_dim)
            C      = np.random.randn(V_null.shape[1], Yj.shape[1])
            Bj     = Bj + 10 * V_null @ C

        dof = n - rank
        return Bj, sse_col, dof

    def svd_L_for_pinv_xtx(Xj: np.ndarray, rcond=1e-12):
        """
        Returns L such that L L^T = (Xj^T Xj)^+  (Moore-Penrose pseudoinverse).
        For Xj = U S V^T (thin SVD), (X^T X)^+ = V diag(1/S^2) V^T,
        so L can be V diag(1/S).
        """
        _, S, Vt = np.linalg.svd(Xj, full_matrices=False)
        if S.size == 0:
            # degenerate: no columns? shouldn't happen for regression
            return np.zeros((Xj.shape[1], 0), dtype=Xj.dtype)

        tol = rcond * S[0]
        mask = S > tol
        V = Vt.T[:, mask]          # (q, r)
        Sinv = 1.0 / S[mask]       # (r,)
        L = V * Sinv[None, :]      # (q, r)  (equivalently V @ diag(Sinv))
        return L

    n, q = X.shape
    _, p = Y.shape

    n_supernodes = len(membership)

    clusters = [[] for _ in range(k)]

    # Build clusters
    for i in range(n):
        clusters[membership[ml_map[i]]].append(i)


    centroids = []
    total_ssr = 0.0

    best_sse = np.inf
    worst_sse = -np.inf
    best_j = 0
    worst_j = 0

    # After building clusters, sort supernodes by their cluster's size (desc)
    # so empty clusters steal from the most populated ones first
    supernode_sizes = (ml_array_idxs[1:] - ml_array_idxs[:-1]) / n

    donor_order = np.random.choice(n_supernodes, size=n_supernodes, replace=False, p=supernode_sizes)
    steal_ptr = 0

    for j, cluster in enumerate(clusters):
        if not cluster:
            chosen = donor_order[steal_ptr]
            steal_ptr += 1
            idx = np.array(ml_array[ml_array_idxs[chosen]:ml_array_idxs[chosen + 1]], dtype=np.int64)
        else:
            idx = np.fromiter(cluster, dtype=np.int64)


        Xj = X[idx, :]
        Yj = Y[idx, :]

        Bj, sse_col, dof = ols_fit_sse(Xj, Yj)

        # Optionally randomize centroid around OLS estimate using column-wise sigma^2
        if random_centroid_scale:
            if dof > 0:
                sigma2 = sse_col / dof                 # (p,)
                L = svd_L_for_pinv_xtx(Xj, rcond=rcond)  # (q, r)
                r = L.shape[1]
                if r > 0:
                    Z = np.random.randn(r, p) * random_centroid_scale
                    Bj = Bj + L @ (np.sqrt(sigma2)[None, :] * Z)

        centroids.append(Bj)

        sse_total = float(np.sum(sse_col))
        if cluster:
            total_ssr += sse_total

        # Only compare "real" clusters that had >= q points originally?
        # Your old logic used remaining_needed < 0; I keep it the same:
        if reposition:
            if sse_total < best_sse:
                best_sse = sse_total
                best_j = j
            if sse_total > worst_sse:
                worst_sse = sse_total
                worst_j = j

    # Distance matrix: SSE for each super-node to each centroid
    XtX = np.empty((n_supernodes, q, q), dtype=X.dtype)
    XtY = np.empty((n_supernodes, q, p), dtype=X.dtype)
    YtY = np.empty(n_supernodes, dtype=X.dtype)

    for i in range(n_supernodes):
        a = ml_array_idxs[i]
        b = ml_array_idxs[i + 1]
        idx = ml_array[a:b]

        Xi = X[idx, :]
        Yi = Y[idx, :]

        XtX[i] = Xi.T @ Xi
        XtY[i] = Xi.T @ Yi
        YtY[i] = np.sum(Yi * Yi)

    # Bstack = np.stack(centroids, axis=0)   # (k, q, p)
    # block = 8192
    # D = np.empty((n_supernodes, k), dtype=np.float64)

    # for s in range(0, n_supernodes, block):
    #     t = min(s + block, n_supernodes)

    #     Gblk = XtX[s:t]   # (b, q, q)
    #     Hblk = XtY[s:t]   # (b, q, p)
    #     cblk = YtY[s:t]   # (b,)

    #     # Only ~3500 einsum calls instead of 100M+
    #     GBblk = np.einsum('iab,kbp->ikap', Gblk, Bstack)
    #     term2 = 2.0 * np.einsum('iqp,kqp->ik', Hblk, Bstack)
    #     term3 = np.einsum('ikqp,kqp->ik', GBblk, Bstack)

    #     D[s:t, :] = cblk[:, None] - term2 + term3

    # term2 = 2.0 * np.einsum('iqp,kqp->ik', XtY, Bstack)
    # GB_all = np.einsum('iab,kbp->ikap', XtX, Bstack)
    # term3 = np.einsum('kqp,ikqp->ik', Bstack, GB_all)

    # D = YtY[:, None] - term2 + term3   
    Bstack = cp.asarray(np.stack(centroids, axis=0), dtype=cp.float64)
    XtX_gpu = cp.asarray(XtX, dtype=cp.float64)
    XtY_gpu = cp.asarray(XtY, dtype=cp.float64)
    YtY_gpu = cp.asarray(YtY, dtype=cp.float64)

    block = 4096
    D_gpu = cp.empty((n_supernodes, k), dtype=cp.float64)

    for s in range(0, n_supernodes, block):
        t = min(s + block, n_supernodes)

        Gblk = XtX_gpu[s:t]
        Hblk = XtY_gpu[s:t]
        cblk = YtY_gpu[s:t]

        GBblk = cp.einsum('iab,kbp->ikap', Gblk, Bstack, optimize=True)
        term2 = 2.0 * cp.einsum('iqp,kqp->ik', Hblk, Bstack, optimize=True)
        term3 = cp.einsum('ikqp,kqp->ik', GBblk, Bstack, optimize=True)

        D_gpu[s:t, :] = cblk[:, None] - term2 + term3

    D = cp.asnumpy(D_gpu)

    if reposition:
        D[:, best_j] = D[:, worst_j]

    return D, round(total_ssr, 5)

def KempeChainMWSP(k, membership, X, Y, sub_adjs, ml_array, ml_array_idxs, ml_map, deepest = False, verbose = True):
    while True:
        D, obj = CentroidUpdate(membership, ml_map, X, Y, k, ml_array, ml_array_idxs, reposition=False, rcond=1e-12, random_centroid_scale=0)
        if verbose:
            print(obj)

        if deepest:
            membership, n_s = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
        else:
            membership, n_s = KSAssignment(k, membership, sub_adjs, D)

        if not n_s:
            return membership

def KempeChainMutation_target_centroids(k, membership, sub_adjs, D):
    n_s = 0
    while True:
        membership, n_s_inner = KSAssignment(k, membership, sub_adjs, D, skip_phases = n_s)
        n_s += n_s_inner
        if not n_s_inner:
            return membership, n_s

def KempeChainMutation_distr(k, membership, X, Y, sub_adjs, ml_array, ml_array_idxs, ml_map, random_centroid_scale):
    D, _ = CentroidUpdate(membership, ml_map, X, Y, k, ml_array, ml_array_idxs, random_centroid_scale = random_centroid_scale)
    membership, _ = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
    return membership

def KSAssignment(k, membership, sub_adjs, D, skip_phases = False):
    def apply_swaps(vertices, membership, chosen, swaps_i_array, swaps_j_array, swaps_Hi_array, swaps_Hj_array, swaps_Hi_index_array, swaps_Hj_index_array):
        cnt = ((swaps_Hi_index_array[chosen + 1] - swaps_Hi_index_array[chosen]).sum()
                + (swaps_Hj_index_array[chosen + 1] - swaps_Hj_index_array[chosen]).sum()
                )
        idx = np.empty(cnt, dtype=vertices.dtype)
        val = np.empty(cnt, dtype=membership.dtype)
        pos = 0
        for sid in chosen:
            i = swaps_i_array[sid]
            j = swaps_j_array[sid]
            a = swaps_Hi_index_array[sid]; b = swaps_Hi_index_array[sid+1]
            hi = swaps_Hi_array[a:b]
            vi = vertices[hi]
            L = vi.size
            idx[pos:pos+L] = vi
            val[pos:pos+L] = j
            pos += L
            a = swaps_Hj_index_array[sid]; b = swaps_Hj_index_array[sid+1]
            hj = swaps_Hj_array[a:b]
            vj = vertices[hj]
            L = vj.size
            idx[pos:pos+L] = vj
            val[pos:pos+L] = i
            pos += L
        membership[idx] = val
    def MWSP(
        weights,
        points_in_swaps_arr,
        points_in_swaps_arr_idx,
        adj_swaps_u_array,
        adj_swaps_v_array,
        time_limit=600,
    ):
        """
        Solve MWSP without Gurobi using scipy.optimize.milp.

        weights : (n,) array
            Objective coefficients.

        points_in_swaps_arr : 2d int array
            Row v contains swap indices involving point v.
            Only the first points_in_swaps_arr_idx[v] entries are valid.

        points_in_swaps_arr_idx : 1d int array
            points_in_swaps_arr_idx[v] = number of valid entries in row v.

        adj_swaps_u_array, adj_swaps_v_array : 1d int arrays
            Conflict edges between swaps.
        """

        weights = np.asarray(weights, dtype=float)
        points_in_swaps_arr = np.asarray(points_in_swaps_arr, dtype=np.int64)
        points_in_swaps_arr_idx = np.asarray(points_in_swaps_arr_idx, dtype=np.int64)
        adj_swaps_u_array = np.asarray(adj_swaps_u_array, dtype=np.int64)
        adj_swaps_v_array = np.asarray(adj_swaps_v_array, dtype=np.int64)

        n_swaps = weights.size

        rows = []
        cols = []
        data = []
        ub = []

        row = 0

        # ------------------------------------------------------------
        # 1. Packing constraints:
        #    For each point v:
        #        sum_{swap containing v} x[swap] <= 1
        # ------------------------------------------------------------
        for v in range(len(points_in_swaps_arr_idx)):
            clique_len = points_in_swaps_arr_idx[v]

            if clique_len > 1:
                clique = points_in_swaps_arr[v, :clique_len]

                # Optional safety filter
                clique = clique[(clique >= 0) & (clique < n_swaps)]

                if len(clique) > 1:
                    rows.extend([row] * len(clique))
                    cols.extend(clique.tolist())
                    data.extend([1.0] * len(clique))
                    ub.append(1.0)
                    row += 1

        # ------------------------------------------------------------
        # 2. Pairwise conflict constraints:
        #        x[u] + x[v] <= 1
        # ------------------------------------------------------------
        mask = (
            (adj_swaps_u_array >= 0)
            & (adj_swaps_u_array < n_swaps)
            & (adj_swaps_v_array >= 0)
            & (adj_swaps_v_array < n_swaps)
            & (adj_swaps_u_array != adj_swaps_v_array)
        )

        us = adj_swaps_u_array[mask]
        vs = adj_swaps_v_array[mask]

        for u, v in zip(us, vs):
            rows.extend([row, row])
            cols.extend([int(u), int(v)])
            data.extend([1.0, 1.0])
            ub.append(1.0)
            row += 1

        # ------------------------------------------------------------
        # If no constraints exist, choose all negative-weight swaps
        # ------------------------------------------------------------
        if row == 0:
            chosen = np.flatnonzero(weights < 0)
            return chosen, None

        A = csr_matrix((data, (rows, cols)), shape=(row, n_swaps))

        constraints = LinearConstraint(
            A,
            lb=-np.inf * np.ones(row),
            ub=np.asarray(ub, dtype=float),
        )

        bounds = Bounds(
            lb=np.zeros(n_swaps),
            ub=np.ones(n_swaps),
        )

        integrality = np.ones(n_swaps, dtype=int)

        res = milp(
            c=weights,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": time_limit,
                "disp": False,
                "presolve": True,
            },
        )

        if res.x is None:
            raise RuntimeError(
                f"MILP failed. Status: {res.status}, message: {res.message}"
            )

        chosen = np.flatnonzero(res.x > 0.5)

        return chosen
    
    n_s = 0
    if not skip_phases:
        membership, n_s = Assign_Singletons_and_Cliques(membership, sub_adjs, D, n_s)
    if k == 2:
        eps = 1e-6
        for vertices, neighbors, neighbors_idx, n_vertices in sub_adjs['others']:
            membership_vertices = membership[vertices]
            color_class_i = []
            color_class_j = []
            for local_idx, c in enumerate(membership_vertices):
                if c:
                    color_class_j.append(local_idx)
                else:
                    color_class_i.append(local_idx)
            i = 0
            j = 1
            D_c = D[vertices, j] - D[vertices, i]
            if not color_class_i:
                for v in color_class_j:
                    delta = -D_c[v]
                    if delta < -eps:
                        membership[vertices[v]] = i
            elif not color_class_j:
                for v in color_class_i:
                    delta = D_c[v]
                    if delta < -eps:
                        membership[vertices[v]] = j
            else:
                visited = np.zeros(n_vertices, dtype=np.bool_)
                q_v = np.empty(n_vertices, dtype=np.int64)
                q_side = np.empty(n_vertices, dtype=np.bool_)
                for start in color_class_i:
                    if visited[start]:
                        continue
                    head = 0
                    tail = 0
                    q_v[tail] = start
                    q_side[tail] = True
                    tail += 1
                    visited[start] = True
                    delta = 0.0
                    while head < tail:
                        v = q_v[head]
                        side = q_side[head]
                        head += 1
                        if side:
                            delta += D_c[v]
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == j) and (not visited[u]):
                                    visited[u] = True
                                    q_v[tail] = u
                                    q_side[tail] = False
                                    tail += 1
                        else:
                            delta -= D_c[v]
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == i) and (not visited[u]):
                                    visited[u] = True
                                    q_v[tail] = u
                                    q_side[tail] = True
                                    tail += 1
                    if delta < -eps:
                        membership[vertices[q_v[:tail]]] = q_side[:tail]
                for v in color_class_j:
                    if not visited[v]:
                        delta = -D_c[v]
                        if delta < -eps:
                            membership[vertices[v]] = i
    else:
        for vertices, neighbors, neighbors_idx, n_vertices in sub_adjs['others']:
            D_vertices  = D[vertices]
            membership_vertices = membership[vertices]
            n_swaps_cap = min(80000, int(n_vertices * (n_vertices - 1)/2))
            swaps_i_array, swaps_j_array, swaps_Hi_array, swaps_Hj_array, swaps_Hi_index_array, swaps_Hj_index_array, swaps_weights_array, points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_array, adj_swaps_v_array = build_MWSP_core(D_vertices, neighbors, neighbors_idx, membership_vertices, k, n_vertices, n_swaps_cap)
            if swaps_i_array.size:
                chosen = MWSP(swaps_weights_array, points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_array, adj_swaps_v_array)
                apply_swaps(vertices, membership, chosen, swaps_i_array, swaps_j_array, swaps_Hi_array, swaps_Hj_array, swaps_Hi_index_array, swaps_Hj_index_array)
                n_s += chosen.size

    return membership, n_s

def KSKM(random_state, X, Y, ml_supernodes, cl_supernodes, steps_mutation, k, steps_back_to_best = 3, steps_no_improvement = 10, verbose = False, time_limit = 3600, membership = None, reposition_frequency = 5, random_centroid_scale = 10, weight_supernodes = False):
    def KSKM_inner(k, membership, X, Y, sub_adjs, ml_array, ml_array_idxs, ml_map, steps_mutation, steps_back_to_best, steps_no_improvement, verbose = True, time_limit = time_limit, reposition_frequency = reposition_frequency, random_centroid_scale = random_centroid_scale, rcond=1e-5):
        start_time = time.time()
        count = 0
        if verbose:
            print('descence')
        membership = KempeChainMWSP(k, membership, X, Y, sub_adjs, ml_array, ml_array_idxs, ml_map, verbose = verbose)
        _, best_obj = CentroidUpdate(membership, ml_map, X, Y, k, ml_array, ml_array_idxs, reposition=False, rcond=1e-12, random_centroid_scale = 0)
        best_membership = copy.deepcopy(membership)
        for i in range(steps_mutation):
            time_used = time.time() - start_time
            if time_used > time_limit:
                return best_membership, best_obj
            if i % reposition_frequency == 0:
                if verbose:
                    print('mutate: reposition')
                D, _ = CentroidUpdate(membership, ml_map, X, Y, k, ml_array, ml_array_idxs, reposition=True, rcond=1e-12, random_centroid_scale = 0)
                membership, _ = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
            else:
                if verbose:
                    print('mutate: perturbation')
                membership = KempeChainMutation_distr(k, membership, X, Y, sub_adjs, ml_array, ml_array_idxs, ml_map, random_centroid_scale = random_centroid_scale)

            if verbose:
                print('descence')
            membership = KempeChainMWSP(k, membership, X, Y, sub_adjs, ml_array, ml_array_idxs, ml_map, verbose = verbose)

            _, obj = CentroidUpdate(membership, ml_map, X, Y, k, ml_array, ml_array_idxs, reposition=False, rcond=1e-12, random_centroid_scale = 0)
            if obj < best_obj - 0.00001:
                best_obj = obj
                if verbose:
                    print('update best obj')
                    print(best_obj)
                count = 0
                best_membership = copy.deepcopy(membership)

            else:
                count += 1
                if count > steps_back_to_best:
                    membership = copy.deepcopy(best_membership)
                    if count > steps_no_improvement:
                        break
            
        return best_membership, best_obj
    start_time = time.time()
    np.random.seed(random_state)
    n = len(X)
    adj, ml_map, n_supernodes, ml_array, ml_array_idxs = preprocessing(n, cl_supernodes, ml_supernodes)
    X = np.column_stack([np.ones(n), X])
    sub_adjs = sub_adj_classification(adj)

    if weight_supernodes:
        supernode_weights = 1.0 / (ml_array_idxs[1:] - ml_array_idxs[:-1])
        sqrt_w = np.sqrt(supernode_weights[ml_map])[:, None]
        X = sqrt_w * X
        Y = sqrt_w * Y
        del supernode_weights, sqrt_w

    del adj, ml_supernodes, cl_supernodes

    print('Preprocessing Done')
    
    if membership is None:
        membership = DSATUR(n_supernodes, sub_adjs, k)
        if not len(membership):
            print('DSATUR solution infeasible')
            return []
        print(membership.max())

    if verbose:
        print('Initialization done')
    time_lefted = time_limit - time.time() + start_time
    membership, _ = KSKM_inner(k, membership, X, Y, sub_adjs, ml_array, ml_array_idxs, ml_map, steps_mutation, steps_back_to_best, steps_no_improvement, verbose, time_limit = time_lefted)
    membership_final = recover_ml_from_membership(membership, ml_map)
    
    return membership_final, membership

def DSATUR(n_supernodes, sub_adjs, k):
    membership = np.zeros(n_supernodes, dtype = np.int64)
    for verts in sub_adjs['cliques']:
        for i, v in enumerate(verts):
            membership[v] = i
    for vertices, neighbors, neighbors_idx, n in sub_adjs['others']:
        # Y is (n,d), Y2 is (n,1), C is (k, d) 
        colors = dsatur_color_numba(neighbors, neighbors_idx, n)
        if max(colors) >= k:
            return []

        for i, c in enumerate(colors):
            v = vertices[i]
            membership[v] = c
    return membership

def sub_adj_classification(adj):
    """
    adj: list of lists (undirected, symmetric adjacency list), nodes 0..n-1

    Returns:
        sub_adjs: dict with keys:
            - 'cliques':    list of np.ndarray (vertex indices of each clique component)
            - 'others':     list of tuples (verts, neighbors, neighbors_idx, n_vertices)
            - 'singletons': np.ndarray of all singleton vertices
    """
    n = len(adj)
    visited = [False] * n

    sub_adjs = {'cliques': [], 'others': []}
    singletons = []

    for s in range(n):
        if visited[s]:
            continue

        # --- 1) BFS to get the connected component starting from s ---
        queue = deque([s])
        visited[s] = True
        vertices = []

        while queue:
            u = queue.popleft()
            vertices.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        n_vertices = len(vertices)

        # --- 2) Singletons ---
        if n_vertices == 1:
            singletons.append(vertices[0])
            continue

        # --- 3) Clique check (using global degrees; safe in undirected components) ---
        # In a clique of size n_vertices, every vertex has degree (n_vertices - 1)
        is_clique = True
        for u in vertices:
            if len(adj[u]) != n_vertices - 1:
                is_clique = False
                break

        verts = np.asarray(vertices, dtype=np.int64)  # shape (m,)

        if is_clique:
            # --- 4a) Clique component ---
            sub_adjs['cliques'].append(verts)
        else:
            # --- 4b) General component: build CSR-like representation ---
            # map global -> local id
            nodes_map = {u: i for i, u in enumerate(vertices)}

            neighbors_idx = np.empty(n_vertices + 1, dtype=np.uint64)
            neighbors_idx[0] = 0
            neighbors = []
            pos = 0

            # IMPORTANT: iterate local vertices in order 0..n_vertices-1
            for local_u, u in enumerate(vertices):
                # all neighbors of u are inside this component (undirected + BFS)
                neigh_local = [nodes_map[v] for v in adj[u]]
                pos += len(neigh_local)
                neighbors.extend(neigh_local)
                neighbors_idx[local_u + 1] = pos

            neighbors = np.asarray(neighbors, dtype=np.int64)

            sub_adjs['others'].append((verts, neighbors, neighbors_idx, n_vertices))

    sub_adjs['singletons'] = np.asarray(singletons, dtype=np.int64)
    return sub_adjs


def preprocessing(n, cl_supernodes, ml_supernodes):
    n_supernodes = len(ml_supernodes)
    ml_array_idxs = np.empty(n_supernodes+1, dtype = np.int64)
    ml_array_idxs[0] = 0
    ml_array = []
    ml_map = np.empty(n, dtype = np.int64)
    for i in range(n_supernodes):
        ml_array_idxs[i + 1] = ml_array_idxs[i] + len(ml_supernodes[i])
        for v in ml_supernodes[i]:
            ml_map[v] = i
        ml_array.extend(ml_supernodes[i])
    ml_array = np.array(ml_array, dtype=np.int64)
    adj =  [set() for _ in range(n_supernodes)]
    for uu, vv in cl_supernodes:
        if uu != vv:
            adj[vv].add(uu)
            adj[uu].add(vv)

    return adj, ml_map, n_supernodes, ml_array, ml_array_idxs

def recover_ml_from_membership(membership, ml_map):
    n = ml_map.size
    membership_final = np.empty(n, dtype=ml_map.dtype)
    for i in range(n):
        membership_final[i] = membership[ml_map[i]]
    return membership_final