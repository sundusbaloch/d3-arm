"""Paper 1 reproduction, part 2: modified Apriori on VAERS 2024 (their Tables 8 and 9).

    python scripts/81_paper1_vaers.py --zip /content/drive/MyDrive/d3arm_data/vaers/2024VAERSData.zip --mem 36g

The paper does not state the minimum support it used for VAERS, so we run a
range of values and report, for each, how many of the paper's top-10 pairs
and top-10 triples we also find in our top 10 by ROR.
Writes data/results/paper1_vaers/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from d3arm import config
from d3arm.paper1.vaers import counts, extract, load_transactions, modified_apriori, overlap_ddi, overlap_main
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--zip", required=True)
p.add_argument("--year", type=int, default=2024)
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--minsup", default="3,10,30,100,300,1000")
args = p.parse_args()

spark = get_spark("paper1-vaers", args.cores, args.mem, shuffle_partitions=32)
out = config.RESULTS_DIR / "paper1_vaers"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
csv_dir = extract(args.zip, config.DATA_DIR / "raw" / "vaers" / str(args.year))
tx, info = load_transactions(spark, csv_dir, args.year)
n, nv, nae, va, vv, vva = counts(tx)
info["counting_seconds"] = round(time.time() - t0, 1)
info["paper"] = {"reports": 51984, "distinct_aes": 5894, "distinct_vaccine_names": 123}

ref8 = pd.read_csv(config.REFERENCE_DIR / "paper1_vaers_table8.csv")
ref9 = pd.read_csv(config.REFERENCE_DIR / "paper1_vaers_table9.csv")
runs = []
for ms in [int(x) for x in args.minsup.split(",")]:
    main, ddi = modified_apriori(n, nv, nae, va, vv, vva, ms)
    top_main = main.sort_values(["ror", "a"], ascending=False).head(10)
    top_ddi = ddi.sort_values(["ror", "a"], ascending=False).head(10)
    runs.append({"minsup": ms, "main_pairs": len(main), "ddi_triples": len(ddi),
                 "pairs_with_infinite_ror": int(np.isinf(main["ror"]).sum()),
                 "table8_overlap_top10": overlap_main(top_main, ref8),
                 "table9_overlap_top10": overlap_ddi(top_ddi, ref9)})
    main.to_csv(out / f"main_minsup{ms}.csv", index=False)
    ddi.to_csv(out / f"ddi_minsup{ms}.csv", index=False)
    print(json.dumps(runs[-1]), flush=True)
info["runs"] = runs
info["seconds"] = round(time.time() - t0, 1)
(out / "summary.json").write_text(json.dumps(info, indent=2, default=int))
for k, v in info.items():
    if k != "runs":
        print(f"{k:34s} {v}")

best = max(runs, key=lambda r: (r["table8_overlap_top10"] + r["table9_overlap_top10"], -r["minsup"]))["minsup"]
main = pd.read_csv(out / f"main_minsup{best}.csv")
ddi = pd.read_csv(out / f"ddi_minsup{best}.csv")
pd.set_option("display.width", 220)
print(f"\nour top 10 vaccine-AE pairs by ROR at minsup = {best}:")
print(main.sort_values(["ror", "a"], ascending=False).head(10)[["v", "ae", "a", "ror", "prr", "lift"]].round(2).to_string(index=False))
print(f"\nour top 10 vaccine-vaccine-AE triples by ROR at minsup = {best}:")
print(ddi.sort_values(["ror", "a"], ascending=False).head(10)[["v1", "v2", "ae", "a", "ror"]].round(2).to_string(index=False))
spark.stop()
