# D3-ARM: Deduplicated, Distributed and Drift-aware Association Rule Mining

CS-895 Big Data Analytics (Fall 2026), semester project.
Sundus Baloch and Mulika Fatima.

We take four recent papers that mine incident reports with association rules,
reproduce them, and then build D3-ARM on top: a PySpark pipeline over all of
FAERS (FDA Adverse Event Reporting System), 2014 Q3 - 2024 Q4, that

1. cleans and deduplicates 16.6 million reports with Paper 2's rule (10.6 million left),
2. mines frequent itemsets with our own SON implementation (Apriori, optionally PCY, inside each partition),
   checked itemset by itemset against Spark MLlib's FP-Growth,
3. counts every drug x AE pair (and drug-drug-AE triple) with >= 5 reports and ranks them by
   ROR / PRR with 95% confidence intervals,
4. removes near-duplicate reports with MinHash-LSH,
5. replays the data quarter by quarter (cumulatively) to get time to signal and stable signals,
   and raises alerts only for never-seen signals with a Bloom filter.

Papers 1 and 2 and the novelty: Sundus Baloch. Papers 3 and 4: Mulika Fatima.

| Paper | What we reproduce | Data |
|---|---|---|
| He, Sun & Tan (2025), *Mathematics* | constrained Apriori for drug-drug interactions, 5 indicators compared by AUC | their simulation code + VAERS 2024 |
| Pisanu et al. (2026), *Eur. J. Clin. Pharmacol.* | cleaning, ROR signals, venlafaxine and sex analyses for esketamine | FAERS 2014Q3-2024Q4 |
| Chakraborty et al. (2026), *Traffic Injury Prevention* | target Apriori (consequent = KA severity), redundant-rule pruning; their Texas CRIS data could not be obtained, so the method is applied to Paper 4's data and to FAERS | Paper 4's AV crashes, FAERS |
| Ding et al. (2025), *J. Intelligent & Connected Vehicles* | Apriori rules for AV crashes (Tables 2-7) | their published data (CA DMV + NHTSA SGO, 1,171 crashes) |

## Repo layout

```
d3arm/                  python package (all the Spark code)
  config.py             paths and the quarter range (data folder: env var D3ARM_DATA)
  spark.py              local SparkSession helper (ANSI mode on, as in Spark 4)
  faers/                download, raw -> parquet, cleaning/dedup, transactions
  mining/               apriori.py (+ PCY), son.py, pairs.py (drug x AE, ROR/PRR), triples.py (drug-drug-AE)
  novelty/              lsh.py (MinHash-LSH), windows.py (quarterly replay), bloom.py (Bloom filter)
  paper1/               simulation.py (port of their R code), vaers.py (VAERS 2024 example)
  paper2/               disproportionality.py (esketamine ROR, venlafaxine, sex)
  crash/                arm.py (rules from itemsets, target consequent, arules-style pruning, mlxtend check), av.py (Paper 4 data)
scripts/                numbered scripts, run them in order
tests/                  fake FAERS sample + unit and end-to-end tests (pytest)
data/reference/         small hand-made files (brand -> ingredient map, the papers' published tables)
docs/                   setup notes for Windows and Colab
```

`data/raw`, `data/parquet`, `data/clean`, `data/logs` and `data/results` are created when you
run the scripts and are not committed (see `.gitignore`).

## Main results (Colab Pro, 8 cores, PySpark 4.0.5)

| Step | Result |
|---|---|
| Cleaning | 16,564,211 raw reports -> 14,198,145 (Paper 2: 14,198,145) -> 10,594,711 after Paper 2's dedup rule (Paper 2: 10,126,485) |
| Paper 2 | 186 esketamine signals (paper 173); all 30 of the paper's top PTs are signals; Spearman 0.995 on their RORs |
| Paper 1 | 357 of 360 simulation AUCs within 0.02 of the paper (mean difference 0.0047); 7.42x faster with Spark |
| SON vs FP-Growth | identical itemsets (109 / 318 / 3,217 at 1% / 0.5% / 0.1%); 106 s vs 4,663 s at 0.1% |
| Drug x AE | 2,213,593 pairs with >= 5 reports, 1,233,587 pass a >= 5 and lower ROR CI > 1, in 91 s |
| MinHash-LSH | 377,371 near-duplicate pairs (recall 40/40 against brute force); 365,733 reports removed |
| Time windows | 1,134,726 stable signals; Bloom filter (1% target) 1.6 MB vs 11.1 MB exact, measured FP 0.155% |
| D3-ARM | 1,050,355 signals (-14.9%), keeping 95.2% of Paper 2's esketamine signals |
| Scaling | 1 -> 8 cores: SON 5.2x, drug x AE 3.9x; time linear in data size |

## Setup

Tested with Python 3.12 + PySpark 3.5.3 + Java 17/21 (laptop, tests) and Python 3.13 +
PySpark 4.0.5 + Java 21 (Google Colab, all results). Then

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
python scripts/83_paper1_scale.py --repo-dir <clone>   # Paper 1's simulation at 10k / 100k / 1M reports (rows resampled)
python scripts/10_paper2_ror.py                # Paper 2: esketamine RORs, venlafaxine and sex analyses
python scripts/20_fpgrowth_baseline.py         # baseline miner: MLlib FPGrowth at several supports
python scripts/30_son.py                       # SON (2 passes, Apriori per partition), checked against FPGrowth
python scripts/31_drug_ae_signals.py           # drug x AE counts at 5 reports, ROR and PRR for every pair
python scripts/32_son_pcy.py                   # SON with PCY inside each partition vs plain Apriori (same itemsets, smaller pair tables?)
python scripts/33_ddi_triples.py               # drug-drug-AE triples on all of FAERS: combination ROR and combination risk ratio
python scripts/40_time_windows.py              # replay quarter by quarter: time to signal, stability, Bloom filter alerts
python scripts/50_lsh_near_duplicates.py       # MinHash-LSH (25 bands x 8 rows) near-duplicate reports, with a recall check
python scripts/51_mass_reporting.py            # reports without age/event date: flag identical-content groups (no removal)
python scripts/60_ablation.py                  # signals after each step: dedup -> + LSH -> + stability over time
python scripts/70_scaling.py                   # 1/2/4/8 cores and 10-100% of the data, for SON and the drug x AE counts
python scripts/91_paper4_av_rules.py          # Paper 4: Tables 2-7 from the authors' own data (SON on Spark + mlxtend check)
python scripts/92_paper3_method_on_av.py      # Paper 3's method on Paper 4's data + consistency check of Paper 3's tables
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

45 tests. The end-to-end test builds two fake FAERS quarters with planted
duplicates (a case with two versions, a deleted case, an exact duplicate pair
and a near-duplicate pair) and runs every FAERS script on them. Unit tests
check Apriori, SON (with and without PCY) and the triple counts against brute
force, LSH recall, the Bloom filter's false-positive rate, the Paper 2
arithmetic against the paper's own numbers, and the Paper 1 port. About 6 minutes.

## Status

- [x] FAERS download, parquet conversion, Paper 2 dedup, transactions
- [x] Paper 1 reproduction: simulation study, VAERS 2024 example, simulation at up to 1M reports
- [x] Paper 2 reproduction: cleaning, ROR, venlafaxine sensitivity, sex-stratified ROR
- [ ] Paper 2 DDI rules (lift, conviction, CRR): need the authors' unpublished drug list; our own
      drug-drug-AE analysis is in scripts/33_ddi_triples.py
- [x] SON (Apriori or PCY per partition, anti-monotone drug/AE constraint), checked against FP-Growth
- [x] drug x AE counts and ROR/PRR for all drugs; drug-drug-AE triples
- [x] MinHash-LSH near-duplicate detection
- [x] time windows (time to signal, stable signals) and Bloom filter alerts
- [x] scaling experiments, ablation, all methods on the same data
- [x] Paper 4 reproduction: Tables 2-4 all 80 rows found exactly; crash-type Tables 5-7 partly (their crash-type input file is not in the package)
- [x] Paper 3: method applied to Paper 4's data; its published tables checked for internal consistency (Texas CRIS data not obtained)
