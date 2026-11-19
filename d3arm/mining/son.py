"""SON (Savasere, Omiecinski & Navathe, 1995) on Spark.

Pass 1 (mapPartitions): each partition holds a fraction p of the baskets and
    runs Apriori in memory with the threshold scaled down to p * s.
    Every itemset that is frequent in at least one partition is a candidate.
    No false negatives: if an itemset had count < p_i * s in EVERY partition,
    its total would be < s, so it cannot be globally frequent.
Between passes: the candidates are collected and broadcast to every worker.
Pass 2 (mapPartitions + reduceByKey): count every candidate over all baskets
    and keep the ones with count >= s. This removes the false positives, so
    the result is exactly the same as Apriori / FP-Growth on the whole data.
"""
import time
from collections import Counter
from operator import add

from pyspark.accumulators import AccumulatorParam

from d3arm.mining.apriori import apriori, count_candidates


class _MaxParam(AccumulatorParam):
    def zero(self, v):
        return 0

    def addInPlace(self, a, b):
        return max(a, b)


def son(baskets, n_total, min_count, allowed=None, max_k=None, pcy_buckets=None):
    """
    baskets     : RDD of sorted item tuples (cache it: it is read twice)
    n_total     : number of baskets
    min_count   : global support threshold, as a count
    pcy_buckets : if set, pass 1 uses PCY (hash buckets) for the pairs in each partition
    Returns ({itemset: count}, stats dict).
    """
    sc = baskets.context
    stats = {}
    pairs_counted = sc.accumulator(0)                  # size of the pair tables in pass 1, summed over partitions
    max_pairs = sc.accumulator(0, _MaxParam())         # largest pair table in one partition (the memory peak)
    freq_buckets = sc.accumulator(0)

    def local_pass(it):
        part = list(it)
        if not part:
            return []
        thr = min_count * len(part) / n_total          # p_i * s
        st = {}
        res = apriori(part, thr, allowed, max_k, pcy_buckets=pcy_buckets, stats=st)
        pairs_counted.add(st.get("pairs_counted", 0))
        max_pairs.add(st.get("pairs_counted", 0))
        freq_buckets.add(st.get("frequent_buckets", 0))
        return list(res.keys())

    t0 = time.time()
    cands = baskets.mapPartitions(local_pass).distinct().collect()
    stats["pass1_seconds"] = round(time.time() - t0, 1)
    stats["candidates"] = len(cands)
    stats["pass1_pairs_counted"] = pairs_counted.value
    stats["pass1_largest_pair_table"] = max_pairs.value
    if pcy_buckets:
        stats["pcy_buckets_per_partition"] = pcy_buckets
        stats["pcy_frequent_buckets"] = freq_buckets.value

    by_k = {}
    for c in cands:
        by_k.setdefault(len(c), set()).add(c)
    stats["candidates_by_size"] = {k: len(v) for k, v in sorted(by_k.items())}
    bc = sc.broadcast(by_k)

    def count_pass(it):
        part = list(it)
        cnt = Counter()
        for k, cs in bc.value.items():
            if k == 1:
                for b in part:
                    for i in b:
                        if (i,) in cs:
                            cnt[(i,)] += 1
            else:
                count_candidates(part, cs, k, cnt)
        return cnt.items()

    t1 = time.time()
    freq = dict(baskets.mapPartitions(count_pass).reduceByKey(add)
                       .filter(lambda kv: kv[1] >= min_count).collect())
    stats["pass2_seconds"] = round(time.time() - t1, 1)
    stats["frequent"] = len(freq)
    stats["false_positives_removed"] = len(cands) - len(freq)
    bc.unpersist()
    return freq, stats
