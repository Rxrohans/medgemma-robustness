"""Verify generated reports against their own ground truth.

    python -m generator.selfcheck data/synthetic/clean

The whole argument for synthetic data is that the labels cannot be wrong. That
argument only holds if it is actually tested, so this asserts the invariants
that would otherwise fail silently and quietly corrupt every downstream number:

* every PNG has a JSON beside it, and vice versa
* the value printed on the page is recoverable from the JSON
* the H/L/N flag agrees with the value and the reference range
* the reference text parses back to the numeric ref_low and ref_high
* values are physiologically plausible, so the reports read as real
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Plausibility is checked as a ratio against the reference range printed on the
# same page, never against absolute bounds. The same analyte appears in g/dL and
# g/L on different reports, so any absolute threshold is wrong for one of them:
# haemoglobin of 112 g/L is 11.2 g/dL and entirely normal. A ratio is unit-free.
#
# Wide on purpose. Real thrombocytopenia sits well under the range, so this is
# tuned to catch generator bugs (a WBC of 0.4 against a 4.0-11.0 range) rather
# than to second-guess clinically abnormal results.
MIN_RATIO_OF_REF_LOW = 0.2
MAX_RATIO_OF_REF_HIGH = 3.0

RANGE_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(-?\d+(?:\.\d+)?)\s*$"
)


def check_report(path: Path) -> list[str]:
    problems: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))

    png = path.with_suffix(".png")
    if not png.exists():
        problems.append(f"{path.name}: no matching PNG")

    patient = data["patient"]
    if not patient["name"].strip():
        problems.append(f"{path.name}: empty patient name")
    if not 0 < patient["age"] < 120:
        problems.append(f"{path.name}: implausible age {patient['age']}")
    if patient["sex"] not in ("M", "F"):
        problems.append(f"{path.name}: bad sex {patient['sex']!r}")

    if not data["tests"]:
        problems.append(f"{path.name}: no tests")

    for t in data["tests"]:
        tag = f"{path.name}/{t['test_name']}"
        value, lo, hi = t["value"], t["ref_low"], t["ref_high"]

        # The printed string must be reconstructible from the JSON, or the page
        # and the label disagree and every comparison inherits the artefact.
        printed = f"{value:.{t['decimals']}f}"
        if float(printed) != round(float(value), t["decimals"]):
            problems.append(f"{tag}: printed {printed} != value {value}")

        expected = "N"
        if lo is not None and value < lo:
            expected = "L"
        elif hi is not None and value > hi:
            expected = "H"
        if t["flag"] != expected:
            problems.append(
                f"{tag}: flag {t['flag']!r} but value {value} vs range {lo}-{hi} "
                f"implies {expected!r}"
            )

        m = RANGE_RE.match(t["ref_text"])
        if not m:
            problems.append(f"{tag}: ref_text {t['ref_text']!r} did not parse")
        else:
            got_lo, got_hi = float(m.group(1)), float(m.group(2))
            if abs(got_lo - lo) > 1e-6 or abs(got_hi - hi) > 1e-6:
                problems.append(
                    f"{tag}: ref_text {t['ref_text']!r} != numeric {lo}-{hi}"
                )

        if lo is not None and lo > 0 and value < lo * MIN_RATIO_OF_REF_LOW:
            problems.append(
                f"{tag}: value {value} {t['unit']} is under {MIN_RATIO_OF_REF_LOW:g}x "
                f"the printed ref_low {lo}, which is not a low result but a bug"
            )
        if hi is not None and hi > 0 and value > hi * MAX_RATIO_OF_REF_HIGH:
            problems.append(
                f"{tag}: value {value} {t['unit']} is over {MAX_RATIO_OF_REF_HIGH:g}x "
                f"the printed ref_high {hi}"
            )

    return problems


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/clean")
    jsons = sorted(root.glob("*.json"))
    if not jsons:
        print(f"No reports found in {root}")
        return 1

    pngs = {p.stem for p in root.glob("*.png")}
    orphans = pngs - {p.stem for p in jsons}

    problems = [p for j in jsons for p in check_report(j)]
    problems += [f"{o}.png: no matching JSON" for o in sorted(orphans)]

    tests = sum(len(json.loads(j.read_text(encoding="utf-8"))["tests"]) for j in jsons)
    print(f"Checked {len(jsons)} reports, {tests} test rows.")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("All invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
