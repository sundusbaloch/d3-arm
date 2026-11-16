# Running on Google Colab

Colab (Pro, High-RAM CPU runtime: 8 cores, ~50 GB RAM) is what we use for the
full FAERS runs.

**Lesson learned:** do not let Spark write straight to Google Drive. Drive is
a network folder; with thousands of small parquet files it gets very slow and
some files end up empty (0 bytes), which breaks the next step. So Spark always
works on Colab's local disk (`/content/d3arm_data`), and Drive only stores
big single files (the code zip and tar backups).

```python
# --- every new session -------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')
import os, time
os.environ["D3ARM_DATA"] = "/content/d3arm_data"          # local disk
!unzip -q -o /content/drive/MyDrive/d3arm_data/d3-arm.zip -d /content/
%cd /content/d3-arm
!mkdir -p /content/d3arm_data
!tar -xf /content/drive/MyDrive/d3arm_data/parquet.tar -C /content/d3arm_data   # raw parquet
!tar -xf /content/drive/MyDrive/d3arm_data/clean.tar -C /content/d3arm_data     # cleaned data + logs
```

First time only (builds those two tar files):

```python
!pytest -q tests/                                          # fake data, ~1-2 min
!python scripts/prepare_all.py --delete-zip --mem 36g      # ~30 min, restartable
!tar -cf /content/drive/MyDrive/d3arm_data/parquet.tar -C /content/d3arm_data parquet
!python -u scripts/03_clean.py --mem 36g --shuffle-partitions 400
!python -u scripts/04_transactions.py --mem 36g
!tar -cf /content/drive/MyDrive/d3arm_data/clean.tar -C /content/d3arm_data clean logs
```

Notes
- Use `--mem` at about 75% of RAM (36g on High-RAM). Pick the CPU runtime,
  not GPU - Spark does not use the GPU here.
- With 8 cores, the core-scaling runs (local[1] -> local[8]) can be done here.
- If Colab disconnects, the local disk is wiped: re-run the "every new
  session" cell, then carry on from the last step that finished.
