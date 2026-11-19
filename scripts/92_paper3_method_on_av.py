"""Paper 3's method (Chakraborty et al. 2026) applied to Paper 4's AV crash data.

    python scripts/92_paper3_method_on_av.py --mem 8g

Paper 3's own data (Texas CRIS 2017-2022, 9,274 driver-ejection crashes) could
not be obtained, so its tables cannot be re-computed. What we do instead:

1. Check Paper 3's published rules for internal consistency. With the consequent
   fixed to KA, lift = confidence / P(KA), and P(KA) is in its Table A1
   (722 / 3,233 restrained, 4,585 / 6,041 unrestrained). Every row of Tables A2 and
   A3 must satisfy this.
2. Apply the same method to the AV crashes, keeping its design:
     - injury crashes only (Paper 3 used KABC crashes): severity minor, moderate or fatality/serious
     - consequent fixed to the severe outcome (KA ~ fatality/serious)
     - two groups mined separately (Paper 3: restrained vs unrestrained;
       here: Level 2 = ADAS vs Level 4 = ADS)
     - support / confidence thresholds of Paper 3 (0.2% / 65% and 3% / 50%), lift > 1,
       arules-style redundant-rule pruning, top 20 by lift.
   0.2% of a few hundred crashes is less than one crash, so the minimum support is
   also floored at --min-crashes (default 7 = 0.2% of Paper 3's 3,233 restrained crashes).
Frequent itemsets come from our SON on Spark. Writes data/results/paper3_on_av/.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from d3arm import config
from d3arm.crash.arm import make_rules, prune_redundant, show
from d3arm.crash.av import baskets, load
from d3arm.mining.son import son
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--min-crashes", type=int, default=7)
p.add_argument("--max-k", type=int, default=6, help="longest itemset (antecedent + consequent)")
args = p.parse_args()

out = config.RESULTS_DIR / "paper3_on_av"
out.mkdir(parents=True, exist_ok=True)
summary = {}

# ---------- 1. Paper 3's tables: lift = confidence / P(KA)
ref = pd.read_csv(config.REFERENCE_DIR / "paper3_tables_A2_A3.csv")
p_ka = {"A2": 722 / 3233, "A3": 4585 / 6041}
ref["lift_from_conf"] = ref.apply(lambda r: r.confidence_pct / 100 / p_ka[r.table], axis=1)
ref["diff"] = (ref["lift_from_conf"] - ref["lift"]).abs()
summary["paper3_internal_check"] = {
    "rows": len(ref), "p_KA_restrained": round(p_ka["A2"], 4), "p_KA_unrestrained": round(p_ka["A3"], 4),
    "max_abs_lift_difference": round(float(ref["diff"].max()), 3),
    "rows_within_0.01": int((ref["diff"] <= 0.01).sum()),
    "restrained_rules_min_crashes": round(0.002 * 3233, 1)}
print("Paper 3 internal check:", summary["paper3_internal_check"])

# ---------- 2. the method on AV injury crashes
spark = get_spark("paper3-on-av", args.cores, args.mem)
sc = spark.sparkContext
df = load()
SEV = "Highest injury severity alleged"
inj = df[df[SEV].isin(["minor injured", "moderate", "fatality/serious"])].copy()
inj["OUTCOME"] = (inj[SEV] == "fatality/serious").map({True: "KA (fatality/serious)", False: None})
inj["group"] = inj["Vehicle type"].map(lambda v: "Level 2 (ADAS)" if str(v).startswith("adas") else "Level 4 (ADS)")
items = [c for c in df.columns if c != SEV] + ["OUTCOME"]
summary["av_injury_crashes"] = {"injury_crashes": len(inj), "severe": int(inj["OUTCOME"].notna().sum()),
                                "by_group": {g: {"crashes": len(x), "severe": int(x["OUTCOME"].notna().sum())}
                                             for g, x in inj.groupby("group")}}
print(json.dumps(summary["av_injury_crashes"], indent=1))
TARGET = "OUTCOME=KA (fatality/serious)"
runs = []
for g, sub in [("all injury crashes", inj)] + list(inj.groupby("group")):
    b = baskets(sub, items)
    n, n_ka = len(b), int(sub["OUTCOME"].notna().sum())
    for s, c in ((0.002, 0.65), (0.03, 0.50)):
        mc = max(math.ceil(s * n - 1e-9), args.min_crashes)
        run = {"group": g, "crashes": n, "severe": n_ka, "support": s, "confidence": c, "min_crashes": mc}
        if n_ka < mc:
            run["note"] = "fewer severe crashes than the minimum support: no rule possible"
            runs.append(run)
            print(run)
            continue
        rdd = sc.parallelize(b, 4).cache()
        freq, _ = son(rdd, n, mc, max_k=args.max_k)
        rdd.unpersist()
        r = make_rules(freq, n, min_conf=c, min_lift=1.0, lift_strict=True, target=TARGET)
        pr = prune_redundant(r)
        run.update({"base_rate": round(n_ka / n, 3), "rules": len(r), "after_pruning": len(pr)})
        runs.append(run)
        print(f"\n{g}: {n} crashes, {n_ka} severe; support {s} (>= {mc} crashes), confidence {c}: "
              f"{len(r)} rules -> {len(pr)} after removing redundant rules")
        if len(pr):
            print(show(pr, 20))
            name = g.split(" (")[0].replace(" ", "_").lower()
            pr.head(20).to_csv(out / f"top20_{name}_s{s}_c{c}.csv", index=False)
summary["runs"] = runs
(out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
spark.stop()
