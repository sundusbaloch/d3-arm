"""Association rules on a table of categorical crash variables (Papers 3 and 4).

A crash record becomes a basket of "variable=value" items, e.g.
    ("Lighting=Dark, not lighted", "Road=Curve", "Speed=70+", "Severity=KA").
Frequent itemsets come from our SON implementation on Spark (exact, the same
as Apriori / FP-Growth). Rules are then built from the itemset counts:

    support(X => Y)    = n(X u Y) / N
    confidence(X => Y) = n(X u Y) / n(X)
    lift(X => Y)       = confidence / (n(Y) / N)

Paper 3 ("target" Apriori, R arules): the consequent is fixed to the outcome
(e.g. Severity=KA), then redundant rules are removed: X => Y is redundant if a
more general rule X' => Y (X' a proper subset of X) has the same or a higher
confidence (arules' is.redundant, i.e. improvement <= 0).
Paper 4 (Python mlxtend): all rules over the confidence and lift thresholds.
"""
from itertools import combinations

import pandas as pd

MISSING = {"", "NA", "N/A", "NONE", "NULL", "UNKNOWN", "NOT REPORTED", "NOT APPLICABLE", "NAN"}


def to_baskets(pdf, columns, missing=MISSING):
    """pandas table -> list of sorted item tuples "column=value"; missing values give no item"""
    out = []
    for row in pdf[columns].itertuples(index=False):
        items = []
        for col, v in zip(columns, row):
            if v is None or (isinstance(v, float) and v != v):
                continue
            s = str(v).strip()
            if s.upper() in missing:
                continue
            items.append(f"{col}={s}")
        out.append(tuple(sorted(set(items))))
    return out


def frequent_itemsets(spark, baskets, min_support, max_k=None, partitions=4, method="son"):
    """{itemset tuple: count} with count >= ceil(min_support * N); method 'son' (Spark) or 'apriori' (in memory)"""
    import math
    from d3arm.mining.apriori import apriori
    n = len(baskets)
    min_count = math.ceil(min_support * n - 1e-9)
    if method == "apriori":
        return apriori(baskets, min_count, max_k=max_k), min_count
    from d3arm.mining.son import son
    rdd = spark.sparkContext.parallelize(baskets, partitions).cache()
    freq, _ = son(rdd, n, min_count, max_k=max_k)
    rdd.unpersist()
    return freq, min_count


def make_rules(freq, n, min_conf=0.0, min_lift=None, lift_strict=False, target=None, max_consequent=1):
    """Rules from frequent itemsets.
    target: item (or prefix such as 'Severity=') that the consequent must be; then the consequent is that one item.
    Otherwise every non-empty consequent of up to max_consequent items.
    Rules with an empty antecedent ({} => Y) are included only for a target (arules does the same)."""
    rows = []
    for items, f in freq.items():
        if target is not None:
            cons_list = [(y,) for y in items if y == target or (target.endswith("=") and y.startswith(target))]
        else:
            if len(items) < 2:
                continue
            cons_list = [c for k in range(1, min(max_consequent, len(items) - 1) + 1) for c in combinations(items, k)]
        for cons in cons_list:
            ante = tuple(i for i in items if i not in cons)
            n_a = freq[ante] if ante else n
            conf = f / n_a
            lift = conf / (freq[cons] / n)
            rows.append((ante, cons, f, f / n, conf, lift))
    r = pd.DataFrame(rows, columns=["antecedent", "consequent", "count", "support", "confidence", "lift"])
    keep = r["confidence"] >= min_conf - 1e-12
    if min_lift is not None:
        keep &= (r["lift"] > min_lift) if lift_strict else (r["lift"] >= min_lift - 1e-12)
    r = r[keep].copy()
    r["k"] = r["antecedent"].map(len) + r["consequent"].map(len)
    return r.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def prune_redundant(rules, all_rules=None):
    """arules is.redundant: drop X => Y if some X' (proper subset of X, the empty set included)
    has conf(X' => Y) >= conf(X => Y). The general rules are looked up in all_rules (default:
    the same table; arules looks them up among the mined rules, which is the same thing
    when no confidence threshold removed them)."""
    ref = all_rules if all_rules is not None else rules
    conf = {(a, c): v for a, c, v in zip(ref["antecedent"], ref["consequent"], ref["confidence"])}
    keep = []
    for a, c, v in zip(rules["antecedent"], rules["consequent"], rules["confidence"]):
        red = any(conf.get((s, c), -1) >= v - 1e-12
                  for k in range(0, len(a)) for s in combinations(a, k))
        keep.append(not red)
    return rules.loc[pd.Series(keep, index=rules.index, dtype=bool)].reset_index(drop=True)


def mlxtend_rules(baskets, min_support, min_conf, min_lift=1.0, max_len=None):
    """the same analysis with mlxtend (Paper 4's library), for comparison"""
    from mlxtend.frequent_patterns import apriori as mx_apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
    fi = mx_apriori(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
    if fi.empty:
        return fi, pd.DataFrame()
    r = association_rules(fi, metric="confidence", min_threshold=min_conf)
    r = r[r["lift"] >= min_lift - 1e-12]
    return fi, r


def show(rules, n=20):
    t = rules.head(n).copy()
    t["antecedent"] = t["antecedent"].map(lambda a: "{" + ", ".join(a) + "}")
    t["consequent"] = t["consequent"].map(lambda a: "{" + ", ".join(a) + "}")
    return t[["antecedent", "consequent", "count", "support", "confidence", "lift"]].round(3).to_string(index=False)
