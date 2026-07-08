"""Judge calibration — correlation math + pass/fail logic, fully offline."""

from islm.eval.calibration import calibrate, pearson
from islm.llm.prompts import JUDGE_DIMENSIONS


def test_pearson_basics():
    assert pearson([1, 2, 3], [1, 2, 3]) == 1.0  # perfect positive
    assert round(pearson([1, 2, 3], [3, 2, 1]), 3) == -1.0  # perfect negative
    assert pearson([1, 1, 1], [1, 2, 3]) is None  # constant -> undefined
    assert pearson([1], [1]) is None  # too few points


def test_calibrate_perfect_agreement_passes():
    rows = [{d: 2 for d in JUDGE_DIMENSIONS}, {d: 1 for d in JUDGE_DIMENSIONS}]
    report = calibrate(rows, rows)  # identical -> trustworthy
    assert report["ok"] is True
    assert all(v["passed"] for v in report["dimensions"].values())


def test_calibrate_flags_low_correlation():
    human = [{d: v for d in JUDGE_DIMENSIONS} for v in (0, 1, 2, 1, 0)]
    # Judge disagrees on spec_adherence (anti-correlated), agrees elsewhere.
    judge = [dict(row) for row in human]
    for row, val in zip(judge, (2, 1, 0, 1, 2), strict=True):
        row["spec_adherence"] = val
    report = calibrate(human, judge)
    assert report["dimensions"]["spec_adherence"]["passed"] is False
    assert report["ok"] is False


def test_calibrate_count_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        calibrate([{d: 1 for d in JUDGE_DIMENSIONS}], [])
