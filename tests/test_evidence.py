"""Check the archived result and reject incomplete or invalid measurements."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report", ROOT / "benchmarks/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture(scope="module")
def evidence():
    manifest = report.verify_archive()
    prompts = json.loads((ROOT / "benchmarks/data/prompts.json").read_text())["prompts"]
    ids = [p["id"] for p in prompts if p["split"] == "dev"]
    rows = report.read_rows(ROOT / "results/fp32/timings.jsonl")
    return rows, ids, manifest


def test_original_result_recomputes(evidence):
    rows, ids, manifest = evidence
    result = report.summarize(rows, ids, 3, manifest["bootstrap"])
    original = json.loads((ROOT / "results/fp32/original_summary.json").read_text())
    report.verify_summary(result, original)
    assert all(result["checks"].values())
    assert result["configs"]["fastTR:30"]["vs_hf"] == pytest.approx(2.36630858450094)
    assert len(rows) == 900 and len(ids) == 60


def test_incomplete_or_duplicate_records_rejected(evidence):
    rows, ids, manifest = evidence
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            report.summarize(bad, ids, 3, manifest["bootstrap"])


def test_mismatch_cannot_produce_success_chart(evidence, tmp_path):
    rows, ids, manifest = evidence
    bad = [{**r, "match": False} if i == 0 else r for i, r in enumerate(rows)]
    result = report.summarize(bad, ids, 3, manifest["bootstrap"])
    with pytest.raises(ValueError, match="validation failed"):
        report.write_report(result, tmp_path, "Invalid test input")
    assert not (tmp_path / "decoding.svg").exists()
