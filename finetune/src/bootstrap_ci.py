"""Paired cluster bootstrap for per-row eval metrics.

Images are resampled, not rows: several questions share one image, so rows
are not independent. Every arm is scored on the same resampled images, so a
difference between two arms has its own (paired) interval.

Usage:
    X = table of one metric, rows x arms (NaN = row not scored)
    point, draws = bootstrap(wmean, X, images)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    diff = draws[:, j] - draws[:, i]          # paired arm j minus arm i
"""

import numpy as np
import pandas as pd


def image_weights(images, n_boot=2000, seed=0, chunk=250):
    """Yield row-weight matrices (<= chunk draws x n_rows). Entry [b, i] is how
    many times row i's image was drawn in bootstrap draw b. Same images +
    same seed -> same draws, so every metric on one row set is paired."""
    codes, uniq = pd.factorize(np.asarray(images))
    rng = np.random.default_rng(seed)
    p = np.full(len(uniq), 1 / len(uniq))
    for start in range(0, n_boot, chunk):
        counts = rng.multinomial(len(uniq), p, size=min(chunk, n_boot - start))
        yield counts[:, codes].astype(float)


def wmean(W, X):
    """Weighted mean per arm, NaN rows skipped. W: draws x rows, X: rows x arms."""
    ok = ~np.isnan(X)
    return (W @ np.where(ok, X, 0)) / (W @ ok)


def wmedian(W, X):
    """Weighted median per arm, NaN rows skipped."""
    out = np.empty((W.shape[0], X.shape[1]))
    for j in range(X.shape[1]):
        keep = ~np.isnan(X[:, j])
        order = np.argsort(X[keep, j])
        vals, cw = X[keep, j][order], np.cumsum(W[:, keep][:, order], axis=1)
        half = cw[:, -1:] / 2
        idx = (cw < half).sum(axis=1)  # first position reaching half the weight
        nxt = np.minimum(idx + 1, len(vals) - 1)
        # exactly half the weight at or below idx: average with the next value,
        # like the ordinary median of an even count
        tie = np.isclose(cw[np.arange(len(cw)), idx], half[:, 0])
        out[:, j] = np.where(tie, (vals[idx] + vals[nxt]) / 2, vals[idx])
    return out


def wtypemean(W, X, types):
    """Mean over groups (e.g. question types) of the weighted mean in each
    group, so every group counts the same however many rows it has."""
    codes, _ = pd.factorize(np.asarray(types))
    T = np.eye(codes.max() + 1)[codes]  # rows x groups one-hot
    ok = ~np.isnan(X)
    out = np.empty((W.shape[0], X.shape[1]))
    for j in range(X.shape[1]):
        s = W @ (T * np.where(ok[:, j], X[:, j], 0)[:, None])
        n = W @ (T * ok[:, j][:, None])
        with np.errstate(invalid="ignore"):
            out[:, j] = np.nanmean(s / n, axis=1)  # a group can vanish from a draw
    return out


def bootstrap(stat, X, images, n_boot=2000, seed=0, **kw):
    """stat(W, X, **kw) -> draws x arms. Returns (point estimate per arm,
    n_boot x arms bootstrap draws)."""
    X = np.asarray(X, dtype=float)
    point = stat(np.ones((1, len(X))), X, **kw)[0]
    draws = np.vstack([stat(W, X, **kw) for W in image_weights(images, n_boot, seed)])
    return point, draws


def boot_p(draws, point):
    """Two-sided bootstrap p-value for "no difference": shift the draws so
    they center on 0 (the null), then count how often a shifted draw is at
    least as far from 0 as the observed difference. Smallest possible value
    is 1/(n_draws + 1)."""
    null = draws - point
    return (1 + np.sum(np.abs(null) >= abs(point))) / (len(draws) + 1)


def holm(pvals):
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def fmt_p(p, prefix=""):
    """'<0.001' or '0.012'; prefix='p' gives 'p<0.001' / 'p=0.012'."""
    if p < 0.001:
        return f"{prefix}<0.001"
    return f"{prefix}{'=' if prefix else ''}{p:.3f}"
