from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
)

CARE_IMPLEMENTATION_VERSION = "encoded-incremental-exact-v2"

def smd_matrix(X, modes):
    X = np.asarray(X, dtype=object)
    modes = np.asarray(modes, dtype=object)
    return np.sum(X[:, None, :] != modes[None, :, :], axis=2)


def assign_nearest(X, modes):
    d = smd_matrix(X, modes)
    labels = np.argmin(d, axis=1)
    return labels, d[np.arange(len(X)), labels]


def global_mode(X):
    X = np.asarray(X, dtype=object)
    mode = []
    for j in range(X.shape[1]):
        values, counts = np.unique(X[:, j], return_counts=True)
        mode.append(values[np.argmax(counts)])
    return np.asarray(mode, dtype=object)


def _mode_1d(values, prefer=None):
    values, counts = np.unique(values, return_counts=True)
    best = values[counts == counts.max()]
    if prefer is not None and prefer in set(best.tolist()):
        return prefer
    return sorted(best.tolist(), key=str)[0]


def modes_from_labels(X, labels, k, previous=None):
    X = np.asarray(X, dtype=object)
    labels = np.asarray(labels)
    modes = np.empty((k, X.shape[1]), dtype=object)

    for h in range(k):
        idx = np.where(labels == h)[0]
        if len(idx) == 0:
            modes[h] = previous[h]
            continue

        for j in range(X.shape[1]):
            old = previous[h, j] if previous is not None else None
            modes[h, j] = _mode_1d(X[idx, j], old)

    return modes


def _repair_empty_clusters(X, labels, modes):
    labels = labels.copy()
    missing = [h for h in range(len(modes)) if not np.any(labels == h)]
    if not missing:
        return labels

    _, dmin = assign_nearest(X, modes)
    order = np.argsort(-dmin)
    moved = set()

    for h in missing:
        for i in order:
            if int(i) in moved:
                continue
            src = labels[i]
            if np.sum(labels == src) > 1:
                labels[i] = h
                moved.add(int(i))
                break

    return labels


def kmodes_fit(X, init_modes, max_iter=100):
    X = np.asarray(X, dtype=object)
    modes = np.asarray(init_modes, dtype=object).copy()
    k = len(modes)
    previous_labels = None

    for iteration in range(1, max_iter + 1):
        labels, _ = assign_nearest(X, modes)
        labels = _repair_empty_clusters(X, labels, modes)

        new_modes = modes_from_labels(X, labels, k, previous=modes)

        if np.array_equal(new_modes, modes):
            modes = new_modes
            break

        if previous_labels is not None and np.array_equal(labels, previous_labels):
            modes = new_modes
            break

        previous_labels = labels.copy()
        modes = new_modes

    labels, dmin = assign_nearest(X, modes)
    labels = _repair_empty_clusters(X, labels, modes)
    modes = modes_from_labels(X, labels, k, previous=modes)
    labels, dmin = assign_nearest(X, modes)

    return {
        "labels": labels,
        "modes": modes,
        "cost": float(dmin.sum()),
        "iterations": iteration,
    }


def density_cao(X):
    X = np.asarray(X, dtype=object)
    n, m = X.shape
    density = np.zeros(n, dtype=float)

    for j in range(m):
        values, counts = np.unique(X[:, j], return_counts=True)
        freq = dict(zip(values, counts / n))
        density += np.asarray([freq[v] for v in X[:, j]], dtype=float)

    return density / m


def cao_init(X, k):
    X = np.asarray(X, dtype=object)
    density = density_cao(X)

    chosen = [int(np.argmax(density))]
    min_dist = np.sum(X != X[chosen[0]], axis=1).astype(float)

    while len(chosen) < k:
        score = density * min_dist
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.sum(X != X[nxt], axis=1))

    return X[chosen].copy()


def random_init(X, k, rng):
    X = np.asarray(X, dtype=object)
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def sample_potential_modes(X, k, rng):
    X = np.asarray(X, dtype=object)
    m = X.shape[1]

    values = []
    probs = []
    for j in range(m):
        v, c = np.unique(X[:, j], return_counts=True)
        values.append(v)
        probs.append(c / c.sum())

    modes = []
    seen = set()
    tries = 0

    while len(modes) < k and tries < 5000:
        z = tuple(rng.choice(values[j], p=probs[j]) for j in range(m))
        tries += 1
        if z not in seen:
            modes.append(z)
            seen.add(z)

    if len(modes) < k:
        for row in X:
            z = tuple(row)
            if z not in seen:
                modes.append(z)
                seen.add(z)
                if len(modes) == k:
                    break

    return np.asarray(modes, dtype=object)


def huang_init(X, k, rng):
    X = np.asarray(X, dtype=object)
    potential = sample_potential_modes(X, k, rng)

    selected = []
    used = set()

    for z in potential:
        dist = np.sum(X != z, axis=1)
        order = np.argsort(dist, kind="stable")
        pick = next(int(i) for i in order if int(i) not in used)
        selected.append(X[pick].copy())
        used.add(pick)

    return np.asarray(selected, dtype=object)


def kpp_init(X, k, rng):
    X = np.asarray(X, dtype=object)
    n = len(X)

    chosen = [int(rng.integers(n))]
    min_dist = np.sum(X != X[chosen[0]], axis=1).astype(float)

    while len(chosen) < k:
        weights = min_dist ** 2
        weights[chosen] = 0

        if weights.sum() <= 0:
            remain = np.setdiff1d(np.arange(n), np.asarray(chosen))
            nxt = int(rng.choice(remain))
        else:
            nxt = int(rng.choice(n, p=weights / weights.sum()))
            while nxt in chosen:
                nxt = int(rng.choice(n, p=weights / weights.sum()))

        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.sum(X != X[nxt], axis=1))

    return X[chosen].copy()


def _bai_density(points, X):
    X = np.asarray(X, dtype=object)
    points = np.atleast_2d(np.asarray(points, dtype=object))
    out = np.zeros(len(points), dtype=float)

    for i, point in enumerate(points):
        out[i] = -np.mean(np.sum(X != point, axis=1))

    return out


def _bai_candidates(X, exemplar):
    X = np.asarray(X, dtype=object)
    m = X.shape[1]
    dist = np.sum(X != exemplar, axis=1)

    candidates = []
    seen = set()

    for radius in range(1, m + 1):
        idx = np.where(dist <= radius)[0]
        if len(idx) == 0:
            continue
        z = tuple(global_mode(X[idx]))
        if z not in seen:
            candidates.append(np.asarray(z, dtype=object))
            seen.add(z)

    return candidates


def bai_init(X, k):
    X = np.asarray(X, dtype=object)
    z = global_mode(X)
    density = _bai_density(X, X)
    centers = []

    for step in range(k):
        if step == 0:
            exemplar_score = density + np.sum(X != z, axis=1)
        else:
            sep = np.min(smd_matrix(X, np.asarray(centers, dtype=object)), axis=1)
            exemplar_score = density + sep

        exemplar = X[int(np.argmax(exemplar_score))]
        candidates = np.asarray(_bai_candidates(X, exemplar), dtype=object)
        cand_density = _bai_density(candidates, X)

        if step == 0:
            score = (
                cand_density
                + np.sum(candidates != z, axis=1)
                - np.sum(candidates != exemplar, axis=1)
            )
        else:
            sep = np.min(
                smd_matrix(candidates, np.asarray(centers, dtype=object)),
                axis=1,
            )
            score = (
                cand_density
                + sep
                - np.sum(candidates != exemplar, axis=1)
            )

        centers.append(candidates[int(np.argmax(score))].copy())

    return np.asarray(centers, dtype=object)


def hr_init(X, k, rng):
    X = np.asarray(X, dtype=object)
    residents = sample_potential_modes(X, k, rng)

    resident_prefs = []
    hospital_ids = set()

    for r in residents:
        dist = np.sum(X != r, axis=1)
        order = np.lexsort((np.arange(len(X)), dist))[:k]
        pref = [int(i) for i in order]
        resident_prefs.append(pref)
        hospital_ids.update(pref)

    hospital_rank = {}
    for h in sorted(hospital_ids):
        residents_here = [r for r in range(k) if h in resident_prefs[r]]
        residents_here.sort(
            key=lambda r: (int(np.sum(residents[r] != X[h])), r)
        )
        hospital_rank[h] = {r: pos for pos, r in enumerate(residents_here)}

    next_pos = [0] * k
    matched_resident = {}
    matched_hospital = {}
    free = list(range(k))

    while free:
        r = free.pop(0)
        if next_pos[r] >= len(resident_prefs[r]):
            continue

        h = resident_prefs[r][next_pos[r]]
        next_pos[r] += 1

        if h not in matched_hospital:
            matched_hospital[h] = r
            matched_resident[r] = h
            continue

        old = matched_hospital[h]
        if hospital_rank[h][r] < hospital_rank[h][old]:
            matched_hospital[h] = r
            matched_resident[r] = h
            matched_resident.pop(old, None)
            free.append(old)
        else:
            free.append(r)

    used = set(matched_resident.values())
    modes = []

    for r in range(k):
        if r in matched_resident:
            h = matched_resident[r]
        else:
            dist = np.sum(X != residents[r], axis=1)
            order = np.argsort(dist, kind="stable")
            h = next(int(i) for i in order if int(i) not in used)
            used.add(h)

        modes.append(X[h].copy())

    return np.asarray(modes, dtype=object)


def _encode_categorical_for_care(X):
    """
    Encode each categorical attribute independently as compact integers.

    Hamming/SMD comparisons are invariant to the integer labels assigned to
    categories, so this changes representation only, not the CARE-KM
    algorithm or any distance value.
    """
    X = np.asarray(X, dtype=object)
    encoded = np.empty(X.shape, dtype=np.int32)

    for j in range(X.shape[1]):
        _, inv = np.unique(
            X[:, j],
            return_inverse=True,
        )
        encoded[:, j] = inv

    return encoded


def _density_cao_encoded(X_encoded):
    """
    Exact Cao density on an integer-coded categorical matrix.
    """
    X_encoded = np.asarray(X_encoded)
    n, m = X_encoded.shape
    density = np.zeros(n, dtype=float)

    for j in range(m):
        counts = np.bincount(X_encoded[:, j])
        density += counts[X_encoded[:, j]] / n

    return density / m


def _smd_matrix_encoded(X_encoded, modes_encoded):
    """
    Simple-matching dissimilarity for integer-coded categorical data.
    """
    X_encoded = np.asarray(X_encoded)
    modes_encoded = np.atleast_2d(
        np.asarray(modes_encoded)
    )

    return np.count_nonzero(
        X_encoded[:, None, :]
        != modes_encoded[None, :, :],
        axis=2,
    )


def _cao_indices_encoded(X_encoded, k, density):
    """
    Cao initialization expressed as row indices of X_encoded.
    Tie handling follows np.argmax exactly as in cao_init().
    """
    X_encoded = np.asarray(X_encoded)
    chosen = [int(np.argmax(density))]

    min_dist = np.count_nonzero(
        X_encoded != X_encoded[chosen[0]],
        axis=1,
    ).astype(float)

    while len(chosen) < k:
        score = density * min_dist
        score[chosen] = -np.inf

        nxt = int(np.argmax(score))
        chosen.append(nxt)

        new_dist = np.count_nonzero(
            X_encoded != X_encoded[nxt],
            axis=1,
        )
        min_dist = np.minimum(
            min_dist,
            new_dist,
        )

    return np.asarray(
        chosen,
        dtype=np.int64,
    )


def initial_cost(X, modes):
    _, dmin = assign_nearest(X, modes)
    return float(dmin.sum())


def care_init(
    X,
    k,
    candidate_budget=8,
    max_exchanges=4,
    candidate_rule="cao",
    min_gain=0.0,
    return_trace=False,
):
    """
    CARE-KM initialization with exact encoded and incremental exchange
    evaluation.

    This implementation preserves the frozen CARE-KM algorithm. It changes
    only the computational representation used internally:

    1. categorical values are integer-coded once;
    2. Cao density is reused instead of recomputed;
    3. the current n-by-k SMD matrix is maintained across accepted exchanges;
    4. a trial replacement evaluates only the candidate-distance column.

    For replacement of mode h by candidate p, the evaluated objective remains

        J(S - {c_h} + {p})
        = sum_i min(d_i,-h, d(x_i, p)),

    which is the global initial-assignment cost used by CARE-KM.
    """
    X = np.asarray(X, dtype=object)
    n = len(X)

    # Representation-only optimization. SMD values are invariant to the
    # integer labels assigned independently within each attribute.
    X_encoded = _encode_categorical_for_care(X)

    # Compute Cao density once and reuse it both for the parent Cao
    # configuration and for CARE candidate prioritization.
    density = _density_cao_encoded(
        X_encoded
    )
    mode_idx = _cao_indices_encoded(
        X_encoded,
        k,
        density,
    )

    # Keep the complete distance matrix for the current seed set. After an
    # accepted one-mode exchange, only the replaced column changes.
    d_current = _smd_matrix_encoded(
        X_encoded,
        X_encoded[mode_idx],
    )

    labels = np.argmin(
        d_current,
        axis=1,
    )
    current_cost = float(
        d_current[
            np.arange(n),
            labels,
        ].sum()
    )

    trace = []

    for step in range(max_exchanges):
        labels = np.argmin(
            d_current,
            axis=1,
        )
        d_first = d_current[
            np.arange(n),
            labels,
        ]

        if k > 1:
            d_second = np.partition(
                d_current,
                1,
                axis=1,
            )[:, 1]
        else:
            d_second = np.full(
                n,
                np.inf,
            )

        best_gain = float(min_gain)
        best = None
        best_distance_column = None

        for h in range(k):
            idx = np.flatnonzero(
                labels == h
            )
            if len(idx) == 0:
                continue

            if k > 1:
                separation = d_second[idx]
            else:
                separation = np.ones(
                    len(idx)
                )

            if candidate_rule == "density":
                score = density[idx]
            elif candidate_rule == "separation":
                score = separation.astype(
                    float
                )
            else:
                score = (
                    density[idx]
                    * separation
                )

            order = np.argsort(
                -score,
                kind="stable",
            )
            cand_idx = idx[
                order[
                    : min(
                        candidate_budget,
                        len(idx),
                    )
                ]
            ]

            if k > 1:
                best_other = np.where(
                    labels == h,
                    d_second,
                    d_first,
                )
            else:
                best_other = np.full(
                    n,
                    np.inf,
                )

            candidate_encoded = (
                X_encoded[cand_idx]
            )
            candidate_dist = (
                _smd_matrix_encoded(
                    X_encoded,
                    candidate_encoded,
                )
            )

            trial_costs = np.minimum(
                best_other[:, None],
                candidate_dist,
            ).sum(
                axis=0
            ).astype(float)

            for pos, ci in enumerate(
                cand_idx
            ):
                # A duplicate
                # row with exactly the same categorical values as the current
                # mode is skipped even if its row index is different.
                if np.array_equal(
                    X_encoded[ci],
                    X_encoded[mode_idx[h]],
                ):
                    continue

                trial_cost = float(
                    trial_costs[pos]
                )
                gain = (
                    current_cost
                    - trial_cost
                )

                # Use stable traversal order and strict best-positive update semantics.
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (
                        h,
                        int(ci),
                        trial_cost,
                    )
                    best_distance_column = (
                        candidate_dist[
                            :,
                            pos,
                        ].copy()
                    )

        if best is None:
            break

        h, row_id, new_cost = best

        mode_idx = mode_idx.copy()
        mode_idx[h] = row_id

        # Exact incremental update: all other current-mode distance columns
        # are unchanged by a one-mode exchange.
        d_current[:, h] = (
            best_distance_column
        )

        trace.append({
            "step": step + 1,
            "cluster": h,
            "row": row_id,
            "gain": float(best_gain),
            "cost_before": float(
                current_cost
            ),
            "cost_after": float(
                new_cost
            ),
        })

        current_cost = new_cost

    modes = X[mode_idx].copy()

    if return_trace:
        return modes, trace

    return modes

def encode_hamming(X):
    X = np.asarray(X, dtype=object)
    encoded = np.empty(X.shape, dtype=np.int32)

    for j in range(X.shape[1]):
        _, inv = np.unique(X[:, j], return_inverse=True)
        encoded[:, j] = inv

    return encoded


def categorical_dbi(X, labels, modes):
    X = np.asarray(X, dtype=object)
    labels = np.asarray(labels)
    modes = np.asarray(modes, dtype=object)
    k = len(modes)

    scatter = np.zeros(k, dtype=float)
    for h in range(k):
        idx = np.where(labels == h)[0]
        if len(idx) == 0:
            return np.inf
        scatter[h] = np.mean(np.sum(X[idx] != modes[h], axis=1))

    center_dist = smd_matrix(modes, modes).astype(float)
    np.fill_diagonal(center_dist, np.inf)

    score = np.empty(k, dtype=float)
    for i in range(k):
        ratio = (scatter[i] + scatter) / center_dist
        ratio[i] = -np.inf
        score[i] = np.max(ratio)

    return float(np.mean(score))


def evaluate_clustering(
    X,
    y,
    result,
    init_modes,
    si_sample_size=2000,
    si_seed=0,
):
    labels = result["labels"]
    modes = result["modes"]

    dbi = categorical_dbi(X, labels, modes)
    encoded = encode_hamming(X)

    if si_sample_size is not None and len(X) > si_sample_size:
        rng = np.random.default_rng(si_seed)
        idx = np.sort(
            rng.choice(len(X), size=si_sample_size, replace=False)
        )
        si = float(
            silhouette_score(
                encoded[idx],
                labels[idx],
                metric="hamming",
            )
        )
    else:
        si = float(silhouette_score(encoded, labels, metric="hamming"))

    sizes = np.bincount(labels, minlength=len(modes)).astype(float)

    row = {
        "initial_cost": initial_cost(X, init_modes),
        "final_cost": result["cost"],
        "dbi": dbi,
        "si": si,
        "iterations": int(result["iterations"]),
        "min_cluster": int(sizes.min()),
        "max_cluster": int(sizes.max()),
        "cluster_cv": float(sizes.std(ddof=0) / sizes.mean()),
    }


    return row


def init_by_name(name, X, k, seed=0, care_config=None):
    rng = np.random.default_rng(seed)
    key = name.lower()

    if key == "random":
        return random_init(X, k, rng)
    if key == "huang":
        return huang_init(X, k, rng)
    if key == "cao":
        return cao_init(X, k)
    if key in {"bai", "bai2012", "bai-2012"}:
        return bai_init(X, k)
    if key in {"k-modes++", "kmodes++", "kpp"}:
        return kpp_init(X, k, rng)
    if key in {"hr-2023", "hr2023", "hr"}:
        return hr_init(X, k, rng)
    if key in {"care-km", "care"}:
        return care_init(X, k, **(care_config or {}))

    raise ValueError(f"Unknown initializer: {name}")


def run_method(
    name,
    X,
    y,
    k,
    seed=0,
    care_config=None,
    si_sample_size=2000,
    si_seed=0,
    max_iter=100,
):
    t0 = time.perf_counter()
    init_modes = init_by_name(
        name,
        X,
        k,
        seed=seed,
        care_config=care_config,
    )
    t1 = time.perf_counter()

    result = kmodes_fit(X, init_modes, max_iter=max_iter)
    t2 = time.perf_counter()

    row = evaluate_clustering(
        X,
        y,
        result,
        init_modes,
        si_sample_size=si_sample_size,
        si_seed=si_seed,
    )

    row.update({
        "method": name,
        "k": k,
        "seed": seed,
        "init_sec": t1 - t0,
        "cluster_sec": t2 - t1,
        "total_sec": t2 - t0,
    })

    return row, init_modes, result
