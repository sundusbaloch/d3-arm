"""Paper 4's AV crash table (California DMV + NHTSA SGO, 1,171 reports).

The authors' spreadsheet has the same category written in several ways
("Proceeding Straight" / "proceeding straight", "Wet" / "wet", "chaning lanes",
"make right turn"). Their published numbers only come out when these are
merged (e.g. Rain => Wet has lift 17.221 = 1171 / 68 only if "Wet" and "wet"
are one item), so we lower-case, trim and fix the typos.
Values that carry no information are dropped as items: unknown, NaN and
"Other, see Narrative" (the paper's tables never show them).
"""
import re
from pathlib import Path

import pandas as pd

from d3arm import config

XLSX = config.REFERENCE_DIR / "paper4" / "av_crashes.xlsx"

# spreadsheet column -> item name used in the paper's tables
COLUMNS = {
    "VEHICLE TYPE": "Vehicle type",
    "SEASON": "SEASON",
    "TIMEOFDAY": "Time of day",
    "Roadway Type": "Roadway type",
    "Roadway Surface": "Roadway surface",
    "Posted Speed Limit (MPH)": "Posted speed limit (MPH)",
    "Lighting": "Lighting",
    "Weather": "Weather",
    "Crash With": "Crash with",
    "Highest Injury Severity Alleged": "Highest injury severity alleged",
    "CP Pre-Crash Movement": "CP pre-crash movement",
    "SV Pre-Crash Movement": "SV pre-crash movement",
    "SV-ACT": "SV-area",
    "ACCIDENT TYPE": "ACCIDENT TYPE",
}
FIX = {"make right turn": "making right turn", "make left turn": "making left turn",
       "make u-turn": "making u-turn", "chaning lanes": "changing lanes",
       "unknow": "unknown", "unkown": "unknown", "passing other vehicle": "passing"}
DROP = {"unknown", "nan", "", "other, see narrative"}


def norm(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]                                   # speed limits 25.0 -> 25
    s = FIX.get(s, s)
    return None if s in DROP else s


def load(path=XLSX):
    """the 'Data source' sheet, normalised; columns renamed to the paper's item names"""
    raw = pd.read_excel(Path(path), sheet_name="Data source")
    df = pd.DataFrame({new: raw[old].map(norm) for old, new in COLUMNS.items()})
    return df.astype(object).where(df.notna(), None)


def baskets(df, columns=None):
    cols = columns or list(df.columns)
    out = []
    for row in df[cols].itertuples(index=False):
        out.append(tuple(sorted(f"{c}={v}" for c, v in zip(cols, row) if v is not None)))
    return out
