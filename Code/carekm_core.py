from __future__ import annotations

import time
from typing import Iterable

import numpy as np
from sklearn.metrics import silhouette_score


CARE_CONFIG = {
    "candidate_rule": "separation",
    "candidate_budget": 16,
    "max_exchanges": 4,
    "min_gain": 0.0,
}

METHODS = [
    "Random",
    "Huang",
    "Cao",
    "Bai",
    "K-Modes++",
    "HR-2023",
    "CARE-KM",
]

STOCHASTIC_METHODS = {"Random", "Huang", "K-Modes++", "HR-2023"}


def smd_matrix(x: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Simple matching dissimilarity untuk semua pasangan data-mode."""
    x = np.asarray(x, dtype=object)
    modes = np.asarray(modes, dtype=object)
    return np.sum(x[:, None, :] != modes[None, :, :], axis=2)


def assign_nearest(x: np.ndarray, modes: np.ndarray):
    dist = smd_matrix(x, modes)
    labels = np.argmin(dist, axis=1)
    return labels, dist[np.arange(len(x)), labels]


def _mode_1d(values: np.ndarray, previous=None):
    values, counts = np.unique(values, return_counts=True)
    tied = values[counts == counts.max()]
    if previous is not None and previous in set(tied.tolist()):
        return previous
    return sorted(tied.tolist(), key=str)[0]


def _modes_from_labels(x, labels, k, previous):
    x = np.asarray(x, dtype=object)
    labels = np.asarray(labels)
    modes = np.empty((k, x.shape[1]), dtype=object)

    for cluster_id in range(k):
        rows = np.where(labels == cluster_id)[0]
        if len(rows) == 0:
            modes[cluster_id] = previous[cluster_id]
            continue
        for col in range(x.shape[1]):
            modes[cluster_id, col] = _mode_1d(
                x[rows, col], previous[cluster_id, col]
            )
    return modes


def _repair_empty_clusters(x, labels, modes):
    labels = labels.copy()
    missing = [c for c in range(len(modes)) if not np.any(labels == c)]
    if not missing:
        return labels

    _, nearest = assign_nearest(x, modes)
    order = np.argsort(-nearest)
    moved = set()

    for target in missing:
        for row in order:
            if int(row) in moved:
                continue
            source = labels[row]
            if np.sum(labels == source) > 1:
                labels[row] = target
                moved.add(int(row))
                break
    return labels


def kmodes_fit(x, init_modes, max_iter=100):
    """K-Modes standar; CARE-KM hanya memasok mode awalnya."""
    x = np.asarray(x, dtype=object)
    modes = np.asarray(init_modes, dtype=object).copy()
    k = len(modes)
    previous_labels = None

    for iteration in range(1, max_iter + 1):
        labels, _ = assign_nearest(x, modes)
        labels = _repair_empty_clusters(x, labels, modes)
        new_modes = _modes_from_labels(x, labels, k, modes)

        if np.array_equal(new_modes, modes):
            modes = new_modes
            break
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            modes = new_modes
            break

        previous_labels = labels.copy()
        modes = new_modes

    labels, _ = assign_nearest(x, modes)
    labels = _repair_empty_clusters(x, labels, modes)
    modes = _modes_from_labels(x, labels, k, modes)
    labels, nearest = assign_nearest(x, modes)

    return {
        "labels": labels,
        "modes": modes,
        "cost": float(nearest.sum()),
        "iterations": int(iteration),
    }


def _global_mode(x):
    return np.asarray([_mode_1d(x[:, j]) for j in range(x.shape[1])], dtype=object)


def _cao_density(x):
    x = np.asarray(x, dtype=object)
    n, m = x.shape
    density = np.zeros(n, dtype=float)
    for j in range(m):
        values, counts = np.unique(x[:, j], return_counts=True)
        freq = dict(zip(values, counts / n))
        density += np.asarray([freq[v] for v in x[:, j]], dtype=float)
    return density / m


def cao_init(x, k):
    x = np.asarray(x, dtype=object)
    density = _cao_density(x)
    chosen = [int(np.argmax(density))]
    min_dist = np.sum(x != x[chosen[0]], axis=1).astype(float)

    while len(chosen) < k:
        score = density * min_dist
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.sum(x != x[nxt], axis=1))
    return x[chosen].copy()


def random_init(x, k, rng):
    idx = rng.choice(len(x), size=k, replace=False)
    return np.asarray(x, dtype=object)[idx].copy()


def _sample_potential_modes(x, k, rng):
    x = np.asarray(x, dtype=object)
    values = []
    probs = []
    for j in range(x.shape[1]):
        v, c = np.unique(x[:, j], return_counts=True)
        values.append(v)
        probs.append(c / c.sum())

    modes = []
    seen = set()
    attempts = 0
    while len(modes) < k and attempts < 5000:
        row = tuple(rng.choice(values[j], p=probs[j]) for j in range(x.shape[1]))
        attempts += 1
        if row not in seen:
            modes.append(row)
            seen.add(row)

    if len(modes) < k:
        for row in x:
            key = tuple(row)
            if key not in seen:
                modes.append(key)
                seen.add(key)
                if len(modes) == k:
                    break
    return np.asarray(modes, dtype=object)


def huang_init(x, k, rng):
    x = np.asarray(x, dtype=object)
    potential = _sample_potential_modes(x, k, rng)
    selected = []
    used = set()

    for mode in potential:
        dist = np.sum(x != mode, axis=1)
        order = np.argsort(dist, kind="stable")
        row = next(int(i) for i in order if int(i) not in used)
        selected.append(x[row].copy())
        used.add(row)
    return np.asarray(selected, dtype=object)


def kpp_init(x, k, rng):
    x = np.asarray(x, dtype=object)
    n = len(x)
    chosen = [int(rng.integers(n))]
    min_dist = np.sum(x != x[chosen[0]], axis=1).astype(float)

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
        min_dist = np.minimum(min_dist, np.sum(x != x[nxt], axis=1))
    return x[chosen].copy()


def _bai_density(points, x):
    points = np.atleast_2d(np.asarray(points, dtype=object))
    return np.asarray([-np.mean(np.sum(np.asarray(x, dtype=object) != p, axis=1)) for p in points])


def _bai_candidates(x, exemplar):
    x = np.asarray(x, dtype=object)
    dist = np.sum(x != exemplar, axis=1)
    candidates = []
    seen = set()
    for radius in range(1, x.shape[1] + 1):
        rows = np.where(dist <= radius)[0]
        if len(rows) == 0:
            continue
        candidate = tuple(_global_mode(x[rows]))
        if candidate not in seen:
            candidates.append(np.asarray(candidate, dtype=object))
            seen.add(candidate)
    return candidates


def bai_init(x, k):
    x = np.asarray(x, dtype=object)
    global_mode = _global_mode(x)
    density = _bai_density(x, x)
    centers = []

    for step in range(k):
        if step == 0:
            exemplar_score = density + np.sum(x != global_mode, axis=1)
        else:
            separation = np.min(smd_matrix(x, np.asarray(centers, dtype=object)), axis=1)
            exemplar_score = density + separation

        exemplar = x[int(np.argmax(exemplar_score))]
        candidates = np.asarray(_bai_candidates(x, exemplar), dtype=object)
        candidate_density = _bai_density(candidates, x)

        if step == 0:
            score = candidate_density + np.sum(candidates != global_mode, axis=1) - np.sum(candidates != exemplar, axis=1)
        else:
            separation = np.min(smd_matrix(candidates, np.asarray(centers, dtype=object)), axis=1)
            score = candidate_density + separation - np.sum(candidates != exemplar, axis=1)
        centers.append(candidates[int(np.argmax(score))].copy())
    return np.asarray(centers, dtype=object)


def hr_init(x, k, rng):
    """Initializer Hospital-Resident yang dipakai sebagai baseline HR-2023."""
    x = np.asarray(x, dtype=object)
    residents = _sample_potential_modes(x, k, rng)

    resident_prefs = []
    hospital_ids = set()
    for resident in residents:
        dist = np.sum(x != resident, axis=1)
        order = np.lexsort((np.arange(len(x)), dist))[:k]
        pref = [int(i) for i in order]
        resident_prefs.append(pref)
        hospital_ids.update(pref)

    hospital_rank = {}
    for hospital in sorted(hospital_ids):
        candidate_residents = [r for r in range(k) if hospital in resident_prefs[r]]
        candidate_residents.sort(key=lambda r: (int(np.sum(residents[r] != x[hospital])), r))
        hospital_rank[hospital] = {r: pos for pos, r in enumerate(candidate_residents)}

    next_pos = [0] * k
    matched_resident = {}
    matched_hospital = {}
    free = list(range(k))

    while free:
        resident = free.pop(0)
        if next_pos[resident] >= len(resident_prefs[resident]):
            continue
        hospital = resident_prefs[resident][next_pos[resident]]
        next_pos[resident] += 1

        if hospital not in matched_hospital:
            matched_hospital[hospital] = resident
            matched_resident[resident] = hospital
            continue

        old = matched_hospital[hospital]
        if hospital_rank[hospital][resident] < hospital_rank[hospital][old]:
            matched_hospital[hospital] = resident
            matched_resident[resident] = hospital
            matched_resident.pop(old, None)
            free.append(old)
        else:
            free.append(resident)

    used = set(matched_resident.values())
    modes = []
    for resident in range(k):
        if resident in matched_resident:
            hospital = matched_resident[resident]
        else:
            dist = np.sum(x != residents[resident], axis=1)
            order = np.argsort(dist, kind="stable")
            hospital = next(int(i) for i in order if int(i) not in used)
            used.add(hospital)
        modes.append(x[hospital].copy())
    return np.asarray(modes, dtype=object)


def _encode_columns(x):
    x = np.asarray(x, dtype=object)
    encoded = np.empty(x.shape, dtype=np.int32)
    for j in range(x.shape[1]):
        _, inv = np.unique(x[:, j], return_inverse=True)
        encoded[:, j] = inv
    return encoded


def _cao_indices_encoded(x, k, density):
    chosen = [int(np.argmax(density))]
    min_dist = np.count_nonzero(x != x[chosen[0]], axis=1).astype(float)
    while len(chosen) < k:
        score = density * min_dist
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.count_nonzero(x != x[nxt], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def care_init(
    x,
    k,
    candidate_budget=16,
    max_exchanges=4,
    candidate_rule="separation",
    min_gain=0.0,
):
    """Refinement satu-mode dari konfigurasi awal Cao."""
    x = np.asarray(x, dtype=object)
    encoded = _encode_columns(x)
    n, m = encoded.shape

    density = np.zeros(n, dtype=float)
    for j in range(m):
        counts = np.bincount(encoded[:, j])
        density += counts[encoded[:, j]] / n
    density /= m

    mode_idx = _cao_indices_encoded(encoded, k, density)
    dist = np.count_nonzero(
        encoded[:, None, :] != encoded[mode_idx][None, :, :], axis=2
    )
    labels = np.argmin(dist, axis=1)
    current_cost = float(dist[np.arange(n), labels].sum())

    for _ in range(max_exchanges):
        labels = np.argmin(dist, axis=1)
        first = dist[np.arange(n), labels]
        second = np.partition(dist, 1, axis=1)[:, 1] if k > 1 else np.full(n, np.inf)

        best_gain = float(min_gain)
        best = None
        best_column = None

        for cluster_id in range(k):
            rows = np.flatnonzero(labels == cluster_id)
            if len(rows) == 0:
                continue

            separation = second[rows] if k > 1 else np.ones(len(rows))
            if candidate_rule == "density":
                score = density[rows]
            elif candidate_rule == "separation":
                score = separation.astype(float)
            else:
                score = density[rows] * separation

            order = np.argsort(-score, kind="stable")
            candidates = rows[order[: min(candidate_budget, len(rows))]]
            best_other = np.where(labels == cluster_id, second, first)

            candidate_dist = np.count_nonzero(
                encoded[:, None, :] != encoded[candidates][None, :, :], axis=2
            )
            trial_cost = np.minimum(best_other[:, None], candidate_dist).sum(axis=0).astype(float)

            for pos, row_id in enumerate(candidates):
                if np.array_equal(encoded[row_id], encoded[mode_idx[cluster_id]]):
                    continue
                gain = current_cost - float(trial_cost[pos])
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (cluster_id, int(row_id), float(trial_cost[pos]))
                    best_column = candidate_dist[:, pos].copy()

        if best is None:
            break

        cluster_id, row_id, new_cost = best
        mode_idx = mode_idx.copy()
        mode_idx[cluster_id] = row_id
        dist[:, cluster_id] = best_column
        current_cost = new_cost

    return x[mode_idx].copy()


def initial_cost(x, modes):
    _, nearest = assign_nearest(x, modes)
    return float(nearest.sum())


def categorical_dbi(x, labels, modes):
    x = np.asarray(x, dtype=object)
    labels = np.asarray(labels)
    modes = np.asarray(modes, dtype=object)
    k = len(modes)

    scatter = np.zeros(k, dtype=float)
    for cluster_id in range(k):
        rows = np.where(labels == cluster_id)[0]
        if len(rows) == 0:
            return np.inf
        scatter[cluster_id] = np.mean(np.sum(x[rows] != modes[cluster_id], axis=1))

    center_dist = smd_matrix(modes, modes).astype(float)
    np.fill_diagonal(center_dist, np.inf)
    scores = np.empty(k, dtype=float)
    for i in range(k):
        ratio = (scatter[i] + scatter) / center_dist[i]
        ratio[i] = -np.inf
        scores[i] = np.max(ratio)
    return float(np.mean(scores))


def evaluate_clustering(x, result, init_modes, si_sample_size=2000, si_seed=0):
    labels = result["labels"]
    modes = result["modes"]
    dbi = categorical_dbi(x, labels, modes)
    encoded = _encode_columns(x)

    if si_sample_size is not None and len(x) > si_sample_size:
        rng = np.random.default_rng(si_seed)
        rows = np.sort(rng.choice(len(x), size=si_sample_size, replace=False))
        si = float(silhouette_score(encoded[rows], labels[rows], metric="hamming"))
    else:
        si = float(silhouette_score(encoded, labels, metric="hamming"))

    sizes = np.bincount(labels, minlength=len(modes)).astype(float)
    return {
        "initial_cost": initial_cost(x, init_modes),
        "final_cost": float(result["cost"]),
        "dbi": dbi,
        "si": si,
        "iterations": int(result["iterations"]),
        "min_cluster": int(sizes.min()),
        "max_cluster": int(sizes.max()),
        "cluster_cv": float(sizes.std(ddof=0) / sizes.mean()),
    }


def init_by_name(name, x, k, seed=0, care_config=None):
    rng = np.random.default_rng(seed)
    key = name.lower()
    if key == "random":
        return random_init(x, k, rng)
    if key == "huang":
        return huang_init(x, k, rng)
    if key == "cao":
        return cao_init(x, k)
    if key == "bai":
        return bai_init(x, k)
    if key in {"k-modes++", "kmodes++"}:
        return kpp_init(x, k, rng)
    if key in {"hr-2023", "hr2023"}:
        return hr_init(x, k, rng)
    if key in {"care-km", "care"}:
        return care_init(x, k, **(care_config or CARE_CONFIG))
    raise ValueError(f"Initializer tidak dikenal: {name}")


def run_method(
    name,
    x,
    k,
    seed=0,
    care_config=None,
    si_sample_size=2000,
    si_seed=0,
    max_iter=100,
):
    start = time.perf_counter()
    init_modes = init_by_name(name, x, k, seed=seed, care_config=care_config)
    after_init = time.perf_counter()
    result = kmodes_fit(x, init_modes, max_iter=max_iter)
    finish = time.perf_counter()

    row = evaluate_clustering(
        x,
        result,
        init_modes,
        si_sample_size=si_sample_size,
        si_seed=si_seed,
    )
    row.update({
        "method": name,
        "k": int(k),
        "seed": int(seed),
        "init_sec": after_init - start,
        "cluster_sec": finish - after_init,
        "total_sec": finish - start,
    })
    return row, init_modes, result
