"""The ROR arithmetic, checked against numbers printed in Paper 2 (Table 2)."""
import math

from d3arm.paper2.disproportionality import ror_ci, ror_table, sex_table

N_ESK, N_CTRL = 7_790, 10_118_302          # paper: esketamine cases, non-esketamine cases


def test_dissociation_matches_paper():
    a, c = 1217, 1713
    ror, lo, hi = ror_ci(a, N_ESK - a, c, N_CTRL - c)
    assert round(ror, 2) == 1093.46           # paper: 1,093.46 (1,012.06-1,181.41)
    assert abs(lo - 1012.06) < 0.5 and abs(hi - 1181.41) < 0.5


def test_nausea_matches_paper():
    a, c = 483, 420_607
    ror, lo, hi = ror_ci(a, N_ESK - a, c, N_CTRL - c)
    assert (round(ror, 2), round(lo, 2), round(hi, 2)) == (1.52, 1.39, 1.67)


def test_zero_cell_gives_nan_not_crash():
    assert math.isnan(ror_ci(5, 100, 0, 1000)[0])


def test_signal_rule():
    t = ror_table({"X": 4, "Y": 50, "Z": 50}, 1000, {"X": 1, "Y": 10, "Z": 50_000}, 1_000_000)
    sig = dict(zip(t["pt"], t["signal"]))
    assert sig == {"X": False, "Y": True, "Z": False}   # X has < 5 cases, Z is not disproportionate


def test_relative_ror_is_ratio_and_symmetric():
    w = ror_table({"X": 40}, 1000, {"X": 100}, 100_000)
    m = ror_table({"X": 10}, 1000, {"X": 100}, 100_000)
    s = sex_table(w, m).iloc[0]
    assert abs(s["rel_ror_w"] - w["ror"][0] / m["ror"][0]) < 1e-9
    assert bool(s["more_in_women"]) and not bool(s["more_in_men"])
