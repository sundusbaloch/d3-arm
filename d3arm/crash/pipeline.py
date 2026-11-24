"""Paper 4's AV crash reports run through the D3-ARM stages (asked for by
Dr. Daud, 24 Nov 2026): prepare -> deduplicate (MinHash-LSH) -> mine (SON and
disproportionality signals) -> track over time (quarterly replay, Bloom filter).

The crash table is turned into the same shapes the FAERS code uses, so the
FAERS functions run unchanged:
  transactions   primaryid = crash id, window = quarter of the incident date,
                 items "DRUG=<field=value>" for the crash circumstances and
                 "AE=<field=value>" for the two outcomes (injury severity, crash type)
  token sets     every descriptive field, including reporting entity, city, state,
                 incident month and time (these identify a crash), for LSH
"""
import pandas as pd

from d3arm.crash.av import COLUMNS, XLSX, norm

OUTCOMES = ["Highest injury severity alleged", "ACCIDENT TYPE"]
# extra fields that identify a crash (not used as items in Paper 4's rules)
ID_FIELDS = {"Reporting Entity": "Reporter", "City": "City", "State": "State",
             "Incident Time (24:00)": "Time", "Roadway Description": "Roadway description", "CP-AREA": "CP-area"}


def load_full(path=XLSX):
    """all fields, normalised; adds id, month (YYYY-MM) and window (YYYYQn)"""
    raw = pd.read_excel(path, sheet_name="Data source")
    df = pd.DataFrame({new: raw[old].map(norm) for old, new in {**COLUMNS, **ID_FIELDS}.items()})
    df = df.astype(object).where(df.notna(), None)
    d = pd.to_datetime(raw["Incident Date"], errors="coerce")
    df.insert(0, "primaryid", raw["id"].astype("int64"))
    df["month"] = d.dt.strftime("%Y-%m")
    df["window"] = d.dt.year.astype("Int64").astype(str) + "Q" + d.dt.quarter.astype("Int64").astype(str)
    return df


def tokens(df):
    """token set per crash (for LSH): every descriptive field + incident month.
    Season, year and time of day are left out because they follow from the date and time."""
    fields = [c for c in list(COLUMNS.values()) + list(ID_FIELDS.values()) if c not in ("SEASON", "Time of day")]
    out = []
    for row in df[["primaryid", "month"] + fields].itertuples(index=False):
        t = {f"{f}={v}" for f, v in zip(fields, row[2:]) if v is not None}
        if row[1]:
            t.add(f"Month={row[1]}")
        out.append((int(row[0]), sorted(t)))
    return out


def transactions(df):
    """rows (primaryid, window, items): circumstances as DRUG= items, outcomes as AE= items"""
    exposures = [c for c in COLUMNS.values() if c not in OUTCOMES]
    rows = []
    for r in df.to_dict("records"):
        items = [f"DRUG={c}={r[c]}" for c in exposures if r[c] is not None]
        items += [f"AE={c}={r[c]}" for c in OUTCOMES if r[c] is not None]
        rows.append((int(r["primaryid"]), r["window"], sorted(set(items))))
    return rows
