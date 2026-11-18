# D3-ARM: Deduplicated, Distributed and Drift-aware Association Rule Mining

CS-895 Big Data Analytics (Fall 2026), semester project.
Sundus Baloch and Mulika Fatima.

We take four recent papers that mine incident reports with association rules,
reproduce each of them in PySpark, and then build D3-ARM on top: a Spark
pipeline that

1. removes near-duplicate FAERS reports with MinHash-LSH,
2. mines frequent itemsets with our own SON implementation (Apriori/PCY inside each partition),
3. ranks drug -> adverse event rules by ROR / PRR with 95% confidence intervals, and
4. mines each FAERS quarter separately and flags new rules with a Bloom filter.

Everything for the novelty runs on one dataset, FAERS (FDA Adverse Event
Reporting System), 2014 Q3 - 2024 Q4. Each paper is first reproduced on its
own data.

| Paper | What we reproduce | Data |
|---|---|---|
| He, Sun & Tan (2025), *Mathematics* | constrained Apriori for drug-drug interactions, 5 indicators compared by AUC | their simulation code + VAERS 2024 |
| Pisanu et al. (2026), *Eur. J. Clin. Pharmacol.* | ROR signals and DDI rules (lift, conviction, CRR) for esketamine | FAERS 2014Q3-2024Q4 |
| Chakraborty et al. (2026), *Traffic Injury Prevention* | Apriori on driver-ejection crashes, belted vs unbelted | Texas CRIS 2017-2022 (FARS fallback) |
| Ding et al. (2025), *J. Intelligent & Connected Vehicles* | Apriori per crash type, Level 2 and Level 4 automation | their public CA DMV + NHTSA data |

## Repo layout

```
d3arm/              python package (all the Spark code)
  config.py         paths and the quarter range
  spark.py          local SparkSession helper
  faers/            download, raw -> parquet, cleaning/dedup, transactions
scripts/            numbered scripts, run them in order
tests/              fake FAERS sample + end-to-end test of the pipeline
data/reference/     small hand-made files (brand -> ingredient map)
docs/               setup notes for Windows and Colab
```

`data/raw`, `data/parquet`, `data/clean` and `data/logs` are created when you
run the scripts and are not committed (see `.gitignore`).

## Setup

Python 3.10-3.12, Java 17, then

```
pip install -r requirements.txt
```

Windows needs a couple of extra steps for Spark to write parquet (winutils),
see `docs/setup_windows.md`. On Colab see `docs/setup_colab.md`.

## Running the FAERS pipeline

```
python scripts/prepare_all.py --delete-zip     # download + unzip + parquet, quarter by quarter
python scripts/03_clean.py                     # dedup with Paper 2's rule + normalisation
python scripts/04_transactions.py              # one basket of items per report
python scripts/80_paper1_simulation.py --repo-dir <clone of TXMUNC/DDI_MDPI_Mathematics_2025>   # Paper 1: their simulation (Tables 2-7), replications in parallel on Spark
python scripts/81_paper1_vaers.py --zip 2024VAERSData.zip    # Paper 1: modified Apriori on VAERS 2024 (Tables 8-9)
python scripts/82_paper1_vaers_grid.py       # Paper 1: which minsup / confidence settings reproduce Tables 8-9
python scripts/10_paper2_ror.py                # Paper 2: esketamine RORs, venlafaxine and sex analyses
python scripts/20_fpgrowth_baseline.py         # baseline miner: MLlib FPGrowth at several supports
python scripts/30_son.py                       # SON (2 passes, Apriori per partition), checked against FPGrowth
python scripts/31_drug_ae_signals.py           # drug x AE counts at 5 reports, ROR and PRR for every pair
python scripts/40_time_windows.py              # replay quarter by quarter: time to signal, stability, Bloom filter alerts
python scripts/50_lsh_near_duplicates.py       # MinHash-LSH (25 bands x 8 rows) near-duplicate reports, with a recall check
python scripts/51_mass_reporting.py            # reports without age/event date: flag identical-content groups (no removal)
python scripts/60_ablation.py                  # signals after each step: dedup -> + LSH -> + stability over time
python scripts/70_scaling.py                   # 1/2/4/8 cores and 10-100% of the data, for SON and the drug x AE counts
python scripts/90_compare_methods.py           # all four papers' methods and ours on the same FAERS data
```

`prepare_all.py` can be stopped and restarted, it skips quarters that are
already done. If the FDA download fails for a quarter, download that zip by
hand from the [FAERS Quarterly Data Extract page](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html),
put it in `data/raw/faers/` and run `scripts/01_download_faers.py --skip-download`
followed by `scripts/02_to_parquet.py --delete-text`.

`03_clean.py` writes the number of reports left after each dedup step to
`data/logs/dedup_counts_<method>.json`. For the 2014Q3-2024Q4 window, Pisanu et al.
report 14,198,145 before and 10,126,485 after deduplication, so those are
the first numbers we check.

There are two dedup rules (`--dedup`):
- `paper2` (default) copies the rule in Pisanu et al.'s public R script
  (github.com/claudiapis/faers_esketamine): reports are split into rows of
  age, sex, PT, drug name, event date, country and indication, and a report is
  dropped if all of its rows already appear in newer reports.
- `exact` is stricter: whole reports must match on every field and on the
  full sets of drugs, reactions and indications.

All scripts take `--cores` (e.g. `--cores 4` runs Spark as `local[4]`) and `--mem`.

## Tests

```
pytest -q tests/
```

This builds two fake FAERS quarters with planted duplicates (a case with two
versions, a deleted case, an exact duplicate pair and a near-duplicate pair)
and runs steps 1-4 on them. Takes under a minute.

## Status

- [x] FAERS download, parquet conversion, Paper 2 dedup, transactions
- [x] Paper 1 reproduction: simulation study and VAERS 2024 example (constrained Apriori + AUC on the simulation)
- [x] Paper 2 reproduction: ROR, venlafaxine sensitivity, sex-stratified ROR
- [ ] Paper 2 reproduction: DDI rules (lift, conviction, CRR)
- [ ] Papers 3 and 4 reproduction
- [x] SON (Apriori per partition, anti-monotone drug/AE constraint)
- [x] drug x AE counts and ROR/PRR for all drugs
- [x] MinHash-LSH near-duplicate detection
- [x] time windows (time to signal, stable signals) and Bloom filter alerts
- [x] scaling experiments and ablation
