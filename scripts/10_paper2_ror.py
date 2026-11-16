"""Paper 2 reproduction, part 1: disproportionality (ROR) for esketamine.

    python scripts/10_paper2_ror.py --mem 36g

Needs data/clean/reports (step 3) and data/parquet/drug + indi (step 2).
Writes to data/results/paper2/:
    summary.json                 our counts next to the paper's
    ror_main.csv                 esketamine vs all other drugs (paper: 173 signals)
    ror_vs_venlafaxine.csv       sensitivity analysis (paper: 72 signals)
    ror_sex.csv                  women vs men relative RORs (paper: 7 women, 16 men)
    table2_comparison.csv        our top-30 PTs against the paper's Table 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from d3arm import config
from d3arm.paper2.disproportionality import PAPER2, compare_top30, run
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--min-a", type=int, default=5, help="minimum esketamine cases for a signal (paper: 5)")
args = p.parse_args()

t0 = time.time()
spark = get_spark("paper2-ror", args.cores, args.mem)
summary, tables = run(spark, config.CLEAN_DIR, config.PARQUET_DIR, args.min_a)

out = config.RESULTS_DIR / "paper2"
out.mkdir(parents=True, exist_ok=True)
for name, df in tables.items():
    df.to_csv(out / f"{name}.csv", index=False)
cmp = compare_top30(tables["ror_main"], config.REFERENCE_DIR / "paper2_table2.csv")
cmp.to_csv(out / "table2_comparison.csv", index=False)
summary["seconds"] = round(time.time() - t0, 1)
(out / "summary.json").write_text(json.dumps({"ours": summary, "paper": PAPER2}, indent=2))

print(f"\n{'':36s}{'ours':>12s}{'paper':>12s}")
for k, v in summary.items():
    if k == "seconds":
        continue
    ref = PAPER2.get(k)
    print(f"{k:36s}{v:>12,}{(f'{ref:,}' if ref is not None else '-'):>12s}")

pd.set_option("display.width", 200, "display.max_columns", 20)
print("\nTop 30 PTs, ours vs paper Table 2:")
print(cmp[["pt", "a_paper", "a_ours", "c_paper", "c_ours", "ror_paper", "ror_ours", "signal"]]
      .round(2).to_string(index=False))
print(f"\ndone in {summary['seconds']}s, results in {out}")
