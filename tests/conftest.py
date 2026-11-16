# lets `pytest` (not only `python -m pytest`) import the d3arm package from the repo root
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
