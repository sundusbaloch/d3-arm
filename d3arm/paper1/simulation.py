"""Reproduction of the simulation study in Paper 1 (He, Sun & Tan, Mathematics 2025,
13, 1710): how well does the "modified Apriori" pick out true drug -> AE
associations (main effects) and drug-drug interactions (DDIs) when there are
latent factors?

Ported from the authors' R code (github.com/TXMUNC/DDI_MDPI_Mathematics_2025,
functions_DDI_manuscript.R and vax_DDI_Simul_code.R). The drug matrix X is
THEIR sampled VAERS 2018 vaccine data (n = 5000 / 10000 reports, m = 10 / 20
vaccines, plus every pairwise product column "vaxi*vaxj").

Data generation, for d = m AEs (their Equations 8-9):
    logit P(y_ij = 1) = beta_0 + x_i' beta_j + z_i' gamma_j
  - beta: 5 main effects and 3 interaction effects, all = 1, all on AE 1;
    which ones depends on the "beta generation setting" 1-4
  - z_i: q = 2 latent factors, gamma = q x d upper-triangular loadings (diag = 1)
      latent = "no_random_effect"  z = 0                       (their LC = 2)
      latent = "random_effect"     z ~ N(0, I), independent of x (LC = 0)
      latent = "confounded"        z = scale(x A + w), A, w ~ N(0, 1)  (LC = 1)
  - beta_0 chosen so that about 10% of all (report, AE) cells are 1

Apriori with minsup = 3, then secondary indicators (Confidence, Lift, PRR,
ROR, RR) for every frequent (drug, AE) pair and (drug, drug, AE) triple.
Pairs/triples that are not frequent get score 0. AUC over the full grid of
possible pairs / triples, true = non-zero beta.

Differences from the R code (stated in the report):
  - beta_0: R uses optimize() on a noisy simulated prevalence; we solve the
    expected prevalence = 0.1 by bisection (deterministic, same target).
  - random numbers: numpy instead of R, so single replications differ;
    we compare the AVERAGE AUC over replications with their tables.
"""
from itertools import combinations

import numpy as np

LATENT = {"no_random_effect": 2, "random_effect": 0, "confounded": 1}   # name -> their LC code
METRICS = ["conf", "lift", "prr", "ror", "rr"]


def load_setting(repo_dir, n, m):
    """their sampled X and effect indices (needs the 'rdata' package)"""
    import rdata
    base = repo_dir / "DDI MDI Mathematics Code"
    x = rdata.read_rda(str(base / f"vax_sim_n{n}_vax{m}.RData"))["vax_sim"]
    idx = rdata.read_rda(str(base / f"vax{m}_effect_idx.RData"))
    names = [str(c) for c in x.coords[x.dims[1]].values]
    return (np.asarray(x.values, dtype=np.float64), names,
            {k: np.asarray(v, dtype=int) for k, v in idx.items()})


def make_beta(names, m, d, setting, idx):
    """beta (p x d) for setting 1 main only, 2 interactions only, 3 independent, 4 hierarchical"""
    p = len(names)
    beta = np.zeros((p, d))
    main = None if setting == 2 else idx["main_idx"]
    inter = None if setting == 1 else (idx["rand_inter_idx"] if setting == 3 else idx["inter_idx"])
    if main is not None:
        beta[main - 1, 0] = 1.0                      # R is 1-based; all effects on AE 1
    if inter is not None:
        beta[m + inter - 1, 0] = 1.0
    return beta


def simulate_ae(x, beta, latent, rng, q=2, target=0.1):
    n, p = x.shape
    d = beta.shape[1]
    z = rng.standard_normal((n, q))
    lc = LATENT[latent]
    if lc == 1:
        a = rng.standard_normal((p, q))
        z = x @ a + z
        z = (z - z.mean(0)) / z.std(0, ddof=1)      # R's scale()
    elif lc == 2:
        z = np.zeros_like(z)
    gamma = np.zeros((q, d))
    for i in range(q):
        gamma[i, i:] = rng.standard_normal(d - i)
    gamma[np.arange(min(q, d)), np.arange(min(q, d))] = 1.0
    eta = x @ beta + z @ gamma
    lo, hi = -10.0, 0.0                              # same search interval as their optimize()
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + eta)))).mean() > target:
            hi = mid
        else:
            lo = mid
    prob = 1 / (1 + np.exp(-((lo + hi) / 2 + eta)))
    return rng.random((n, d)) < prob


def _scores(a, b, c, d_, n):
    """Confidence, Lift, PRR, ROR, RR from 2x2 cells (arrays); same formulas as their R code"""
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = a / (a + b)
        out = {"conf": conf,
               "lift": conf / ((a + c) / n),
               "prr": (a / (a + b)) / (c / (c + d_)),
               "ror": (a * d_) / (c * b),
               "rr": n * a / ((a + c) * (a + b))}
    return out


def _clean(s, keep):
    s = np.where(keep, s, 0.0)
    s = np.where(np.isnan(s), 0.0, s)                # R: merged[is.na(merged)] <- 0
    return np.where(np.isinf(s), 1e6, s)             # R: Inf -> 1e6


def auc(truth, score):
    """area under the ROC curve (ties count 1/2), = P(score of a true signal > score of a non-signal)"""
    truth = np.asarray(truth, bool).ravel()
    score = np.asarray(score, float).ravel()
    n1, n0 = truth.sum(), (~truth).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                                # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def apriori_scores(v, y, minsup=3):
    """v: n x m drug indicators, y: n x d AE indicators.
    Returns main scores (m x d per metric) and DDI scores ((m choose 2) x d per metric)."""
    n, m = v.shape
    d = y.shape[1]
    v = v.astype(np.int64)
    y = y.astype(np.int64)
    nv, ny = v.sum(0), y.sum(0)
    fv, fy = nv >= minsup, ny >= minsup
    a = v.T @ y                                      # drug x AE co-occurrence counts
    keep = fv[:, None] & fy[None, :] & (a >= minsup)
    s = _scores(a, nv[:, None] - a, ny[None, :] - a, n - nv[:, None] - ny[None, :] + a, n)
    main = {k: _clean(val, keep) for k, val in s.items()}

    pairs = list(combinations(range(m), 2))
    ddi = {k: np.zeros((len(pairs), d)) for k in METRICS}
    vv = v.T @ v                                     # drug x drug counts
    for r, (i, j) in enumerate(pairs):
        if not (fv[i] and fv[j] and vv[i, j] >= minsup):
            continue
        w = v[:, i] * v[:, j]                        # reports with both drugs
        a3 = w @ y
        nw = w.sum()
        keep3 = fy & (a3 >= minsup)
        b3, c3 = nw - a3, ny - a3
        s3 = _scores(a3, b3, c3, n - ny - b3, n)
        for k in METRICS:
            ddi[k][r] = _clean(s3[k], keep3)
    return main, ddi, pairs


def one_replication(x, names, m, beta, latent, seed, minsup=3):
    rng = np.random.default_rng(seed)
    y = simulate_ae(x, beta, latent, rng)
    main, ddi, pairs = apriori_scores(x[:, :m] > 0, y, minsup)
    true_main = beta[:m] != 0
    col = {nm: k for k, nm in enumerate(names)}
    true_ddi = np.stack([beta[col[f"vax{i + 1}*vax{j + 1}"]] != 0 for i, j in pairs])
    return ({k: auc(true_main, main[k]) for k in METRICS},
            {k: auc(true_ddi, ddi[k]) for k in METRICS})


# ---------------------------------------------------------------- larger n (scaling study)
# Their drug matrices have n = 5,000 or 10,000 reports. To study n up to 1,000,000
# we resample their rows with replacement (idx), so the drug exposure pattern
# stays theirs. Nothing n x p is ever built: x[idx] @ beta = (x @ beta)[idx].

def simulate_ae_resampled(x, idx, beta, latent, rng, q=2, target=0.1):
    """Same model and the same order of random draws as simulate_ae, on the rows x[idx].
    simulate_ae_resampled(x, arange(n), ...) == simulate_ae(x, ...) for the same rng state."""
    n, p = len(idx), x.shape[1]
    d = beta.shape[1]
    z = rng.standard_normal((n, q))
    lc = LATENT[latent]
    if lc == 1:
        a = rng.standard_normal((p, q))
        z = (x @ a)[idx] + z
        z = (z - z.mean(0)) / z.std(0, ddof=1)
    elif lc == 2:
        z = np.zeros_like(z)
    gamma = np.zeros((q, d))
    for i in range(q):
        gamma[i, i:] = rng.standard_normal(d - i)
    gamma[np.arange(min(q, d)), np.arange(min(q, d))] = 1.0
    eta = (x @ beta)[idx] + z @ gamma
    lo, hi = -10.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + eta)))).mean() > target:
            hi = mid
        else:
            lo = mid
    prob = 1 / (1 + np.exp(-((lo + hi) / 2 + eta)))
    return rng.random((n, d)) < prob


def apriori_scores_fast(v, y, minsup=3, chunk=32):
    """apriori_scores with matrix products (float32 counts are exact below 2^24 = 16.7M reports),
    the DDI pairs done `chunk` at a time. Same output as apriori_scores."""
    n, m = v.shape
    d = y.shape[1]
    if n >= 2 ** 24:
        raise ValueError("float32 counts are exact only below 16.7M reports")
    vf, yf = v.astype(np.float32), y.astype(np.float32)
    nv, ny = v.sum(0).astype(np.int64), y.sum(0).astype(np.int64)
    fv, fy = nv >= minsup, ny >= minsup
    a = np.rint(vf.T @ yf).astype(np.int64)
    keep = fv[:, None] & fy[None, :] & (a >= minsup)
    s = _scores(a, nv[:, None] - a, ny[None, :] - a, n - nv[:, None] - ny[None, :] + a, n)
    main = {k: _clean(val, keep) for k, val in s.items()}

    pairs = list(combinations(range(m), 2))
    ddi = {k: np.zeros((len(pairs), d)) for k in METRICS}
    vv = np.rint(vf.T @ vf).astype(np.int64)
    todo = [r for r, (i, j) in enumerate(pairs) if fv[i] and fv[j] and vv[i, j] >= minsup]
    for start in range(0, len(todo), chunk):
        rows = todo[start:start + chunk]
        ii = np.array([pairs[r][0] for r in rows])
        jj = np.array([pairs[r][1] for r in rows])
        w = vf[:, ii] * vf[:, jj]                      # n x chunk: reports with both drugs
        a3 = np.rint(w.T @ yf).astype(np.int64)        # chunk x d
        nw = vv[ii, jj][:, None]
        b3, c3 = nw - a3, ny[None, :] - a3
        s3 = _scores(a3, b3, c3, n - ny[None, :] - b3, n)
        keep3 = fy[None, :] & (a3 >= minsup)
        for k in METRICS:
            ddi[k][rows] = _clean(s3[k], keep3)
    return main, ddi, pairs


def one_replication_resampled(x, names, m, beta, latent, seed, n, minsup=3):
    rng = np.random.default_rng(seed)
    idx = np.random.default_rng(seed + 987_654_321).integers(0, x.shape[0], n)
    y = simulate_ae_resampled(x, idx, beta, latent, rng)
    main, ddi, pairs = apriori_scores_fast(x[idx, :m] > 0, y, minsup)
    true_main = beta[:m] != 0
    col = {nm: k for k, nm in enumerate(names)}
    true_ddi = np.stack([beta[col[f"vax{i + 1}*vax{j + 1}"]] != 0 for i, j in pairs])
    return ({k: auc(true_main, main[k]) for k in METRICS},
            {k: auc(true_ddi, ddi[k]) for k in METRICS})
