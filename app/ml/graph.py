"""Graph utilities. Correlation graphs are forecasting supports, never road routes."""
from __future__ import annotations
import numpy as np
import warnings


def correlation_adjacency(training_values: np.ndarray, top_k: int = 8, threshold: float = .3) -> np.ndarray:
    x = np.asarray(training_values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("training_values must be [time, sensors].")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    n = corr.shape[0]; a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        candidates = np.flatnonzero(corr[i] >= threshold)
        if len(candidates):
            chosen = candidates[np.argsort(corr[i, candidates])[-top_k:]]
            a[i, chosen] = corr[i, chosen]
    return np.maximum(a, a.T)


def road_adjacency(sensor_ids: list[str], edges: list[dict]) -> np.ndarray:
    """Create directed DCRNN supports from explicitly supplied road edges."""
    ids={str(sensor):i for i,sensor in enumerate(sensor_ids)}
    a=np.zeros((len(ids),len(ids)),dtype=np.float32)
    for edge in edges:
        source=str(edge["source_sensor"]); target=str(edge["destination_sensor"])
        if source not in ids or target not in ids:
            raise ValueError(f"Road edge references sensor absent from dataset: {source} or {target}.")
        distance=edge.get("edge_distance_km")
        weight=1.0/float(distance) if distance is not None else 1.0
        if not np.isfinite(weight) or weight<=0: raise ValueError("Road graph weights/distances must be positive and finite.")
        a[ids[source],ids[target]]=weight
    return a


def diffusion_supports(adjacency: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(adjacency, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Adjacency must be square.")
    out = a.sum(axis=1); inn = a.sum(axis=0)
    pf = np.divide(a, out[:, None], out=np.zeros_like(a), where=out[:, None] > 0)
    pb = np.divide(a.T, inn[:, None], out=np.zeros_like(a), where=inn[:, None] > 0)
    f = [np.eye(len(a), dtype=np.float32)]; b = [np.eye(len(a), dtype=np.float32)]
    for _ in range(order):
        f.append(f[-1] @ pf); b.append(b[-1] @ pb)
    return np.stack(f), np.stack(b)

