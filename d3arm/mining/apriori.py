"""In-memory Apriori, used inside each Spark partition by SON (pass 1).

Baskets are tuples of item strings, sorted and without repeats. Itemsets are
sorted tuples as well, so ("AE=NAUSEA", "DRUG=ASPIRIN") is always written the
same way and can be used as a dictionary key.

`allowed` is an optional constraint. It must be anti-monotone: if an itemset
is allowed, all of its subsets must be allowed too (e.g. "at most one AE").
Then we can apply it while generating candidates, exactly like the support
check, and never build the itemsets we do not want.
"""
from collections import Counter
from itertools import combinations
from math import comb


def drug_ae_constraint(max_drugs=2):
    """at most one AE= item and at most `max_drugs` DRUG= items"""
    def ok(itemset):
        n_ae = sum(1 for i in itemset if i.startswith("AE="))
        return n_ae <= 1 and len(itemset) - n_ae <= max_drugs
    return ok


def _candidates(prev, k, allowed):
    """Apriori join + prune: size-k candidates whose (k-1)-subsets are all frequent."""
    prev = sorted(prev)
    prev_set = set(prev)
    out = set()
    for i in range(len(prev)):
        a = prev[i]
        for j in range(i + 1, len(prev)):
            b = prev[j]
            if a[:-1] != b[:-1]:
                break                      # sorted, so no later b shares the prefix either
            c = a + (b[-1],)
            if allowed and not allowed(c):
                continue
            if all(c[:x] + c[x + 1:] in prev_set for x in range(k)):
                out.add(c)
    return out


def count_candidates(baskets, cands, k, counter):
    """add to `counter` how many baskets contain each size-k candidate"""
    items = {i for c in cands for i in c}
    for b in baskets:
        bi = [i for i in b if i in items]
        if len(bi) < k:
            continue
        if comb(len(bi), k) <= len(cands):        # few subsets: enumerate them
            for s in combinations(bi, k):
                if s in cands:
                    counter[s] += 1
        else:                                     # huge basket: check candidates instead
            bs = set(bi)
            for c in cands:
                if all(i in bs for i in c):
                    counter[c] += 1


def pcy_bucket_counts(baskets, n_buckets):
    """PCY pass 1 extra: hash EVERY pair of every basket into one of n_buckets and count.
    Items get integer ids (order of first appearance); bucket(i, j) = (i * P + j) mod n_buckets.
    Returns (item -> id, bucket counts). Vectorised per basket with numpy."""
    import numpy as np
    ids, counts, P = {}, np.zeros(n_buckets, dtype=np.int64), 1_000_003
    for b in baskets:
        if len(b) < 2:
            for i in b:
                ids.setdefault(i, len(ids))
            continue
        x = np.fromiter((ids.setdefault(i, len(ids)) for i in b), dtype=np.int64, count=len(b))
        ii, jj = np.triu_indices(len(x), 1)
        lo, hi = np.minimum(x[ii], x[jj]), np.maximum(x[ii], x[jj])
        np.add.at(counts, (lo * P + hi) % n_buckets, 1)
    return ids, counts


def apriori(baskets, min_count, allowed=None, max_k=None, pcy_buckets=None, stats=None):
    """All itemsets with count >= min_count (min_count may be a fraction). Returns {itemset: count}.

    pcy_buckets: if set, use PCY for the pairs: while counting single items, also
    hash every pair into pcy_buckets buckets; a pair is then counted only if both
    items are frequent AND its bucket is frequent. Same result (a frequent pair
    always lies in a frequent bucket), but fewer pairs held in memory.
    stats: optional dict, filled with 'pairs_counted' (size of the pair table) and,
    with PCY, 'frequent_buckets'.
    """
    singles = Counter(i for b in baskets for i in b)
    bitmap = None
    if pcy_buckets:
        ids, bucket_counts = pcy_bucket_counts(baskets, pcy_buckets)
        bitmap = bucket_counts >= min_count                  # 1 bit per bucket in the textbook version
        if stats is not None:
            stats["frequent_buckets"] = int(bitmap.sum())
        bitmap = bitmap.tolist()                             # plain list: fast lookups in the loop below
    level = {(i,): n for i, n in singles.items()
             if n >= min_count and (allowed is None or allowed((i,)))}
    result = dict(level)
    k = 1
    while level and (max_k is None or k < max_k):
        k += 1
        if k == 2:
            # pairs: just count every pair of frequent items that occurs (no need to list candidates)
            freq_items = {c[0] for c in level}
            cnt = Counter()
            for b in baskets:
                fb = [i for i in b if i in freq_items]
                if bitmap is None:
                    for s in combinations(fb, 2):
                        cnt[s] += 1
                else:                                        # PCY: skip pairs in infrequent buckets
                    for s in combinations(fb, 2):
                        x, y = ids[s[0]], ids[s[1]]
                        if bitmap[(min(x, y) * 1_000_003 + max(x, y)) % pcy_buckets]:
                            cnt[s] += 1
            if stats is not None:
                stats["pairs_counted"] = len(cnt)
            level = {s: n for s, n in cnt.items()
                     if n >= min_count and (allowed is None or allowed(s))}
        else:
            cands = _candidates(level.keys(), k, allowed)
            if not cands:
                break
            cnt = Counter()
            count_candidates(baskets, cands, k, cnt)
            level = {s: n for s, n in cnt.items() if n >= min_count}
        result.update(level)
    return result
