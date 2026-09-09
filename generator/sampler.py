"""Realistic CBC value sampling.

Clinical validity is not the goal; an extraction test does not need it. What it
does need is the messiness real reports have, because that messiness is what the
model actually has to cope with:

* Synonyms. SGPT/ALT is the famous pair, but CBC has its own: TLC vs WBC,
  Haemoglobin vs Hb. Labs genuinely disagree, so the model must copy what is
  printed rather than normalise to a favourite spelling.
* Unit variants. Indian labs report WBC as ``/cumm`` where others use
  ``10^3/uL``, and platelets in ``lakhs/cumm``. Same quantity, different number
  on the page.
* Reference-range formats. Hyphen, en dash and spacing all vary.
* Out-of-range values, so the H/L flag column is actually exercised.

Everything is driven by a seeded ``random.Random`` so a benchmark run is
reproducible. A sweep is worthless if the underlying reports drift between runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from .schema import Patient, Report, ReportMeta, Test


@dataclass(frozen=True)
class UnitVariant:
    """One way a lab may print a quantity. ``factor`` scales the canonical value."""

    unit: str
    factor: float
    decimals: int


@dataclass(frozen=True)
class TestSpec:
    canonical: str
    names: tuple[str, ...]              # printed name variants
    units: tuple[UnitVariant, ...]
    ref_male: tuple[float, float]       # in canonical units
    ref_female: tuple[float, float]
    spread: float                       # sampling width beyond the reference range
    panel: str = "CBC"

    def ref_for(self, sex: str) -> tuple[float, float]:
        return self.ref_male if sex == "M" else self.ref_female


# Canonical units are the ones the reference ranges below are expressed in.
CBC_TESTS: tuple[TestSpec, ...] = (
    TestSpec(
        canonical="Hemoglobin",
        names=("Hemoglobin", "Haemoglobin", "Hb", "Hemoglobin (Hb)"),
        units=(UnitVariant("g/dL", 1.0, 1), UnitVariant("g/dl", 1.0, 1),
               UnitVariant("g/L", 10.0, 0)),
        ref_male=(13.0, 17.0), ref_female=(12.0, 15.0), spread=3.0,
    ),
    TestSpec(
        canonical="RBC Count",
        names=("RBC Count", "Total RBC Count", "Erythrocyte Count", "R.B.C."),
        units=(UnitVariant("million/cumm", 1.0, 2), UnitVariant("10^6/uL", 1.0, 2),
               UnitVariant("x10^12/L", 1.0, 2)),
        ref_male=(4.5, 5.9), ref_female=(4.1, 5.1), spread=0.9,
    ),
    TestSpec(
        canonical="WBC Count",
        names=("WBC Count", "Total Leucocyte Count (TLC)", "TLC", "Total WBC Count"),
        units=(UnitVariant("/cumm", 1000.0, 0), UnitVariant("10^3/uL", 1.0, 1),
               UnitVariant("cells/cumm", 1000.0, 0)),
        ref_male=(4.0, 11.0), ref_female=(4.0, 11.0), spread=5.0,
    ),
    TestSpec(
        canonical="Platelet Count",
        names=("Platelet Count", "Platelets", "PLT", "Thrombocyte Count"),
        units=(UnitVariant("/cumm", 1000.0, 0), UnitVariant("10^3/uL", 1.0, 0),
               UnitVariant("lakhs/cumm", 0.01, 2)),
        ref_male=(150.0, 410.0), ref_female=(150.0, 410.0), spread=120.0,
    ),
    TestSpec(
        canonical="MCV",
        names=("MCV", "Mean Corpuscular Volume", "M.C.V."),
        units=(UnitVariant("fL", 1.0, 1), UnitVariant("fl", 1.0, 1)),
        ref_male=(80.0, 100.0), ref_female=(80.0, 100.0), spread=14.0,
    ),
    TestSpec(
        canonical="MCH",
        names=("MCH", "Mean Corpuscular Hemoglobin", "M.C.H."),
        units=(UnitVariant("pg", 1.0, 1),),
        ref_male=(27.0, 33.0), ref_female=(27.0, 33.0), spread=5.0,
    ),
    TestSpec(
        canonical="MCHC",
        names=("MCHC", "Mean Corpuscular Hb Concentration", "M.C.H.C."),
        units=(UnitVariant("g/dL", 1.0, 1), UnitVariant("%", 1.0, 1)),
        ref_male=(32.0, 36.0), ref_female=(32.0, 36.0), spread=3.5,
    ),
)

# Range separators labs actually print. The en dash in the third entry is
# deliberate: it is the character that trips up naive parsers, so it belongs in
# the test data. Do not "clean it up".
RANGE_FORMATS = ("{lo} - {hi}", "{lo}-{hi}", "{lo} – {hi}", "{lo} to {hi}")

# Split by sex so the printed name never contradicts the printed sex field. A
# mismatch would be an artefact no real report has, and one a model could learn
# to key on.
MALE_NAMES = ("Rohan", "Arjun", "Vikram", "Kabir", "Rahul", "Aditya", "Karthik",
              "Sanjay", "Nikhil", "Harsh")
FEMALE_NAMES = ("Priya", "Sneha", "Ananya", "Meera", "Divya", "Nisha", "Pooja",
                "Kavya", "Anjali", "Ritika")
LAST_NAMES = ("Sharma", "Patel", "Reddy", "Nair", "Gupta", "Iyer", "Singh",
              "Mehta", "Das", "Kulkarni", "Banerjee", "Rao")
LAB_NAMES = ("Sunrise Diagnostics", "Apex Path Labs", "City Care Laboratory",
             "Metro Diagnostic Centre", "LifeLine Pathology", "Prime Health Labs")


def _round(value: float, decimals: int) -> float:
    return round(value, decimals) if decimals else float(round(value))


def _sample_value(rng: random.Random, spec: TestSpec, sex: str) -> tuple[float, str]:
    """Return a canonical-unit value and its H/L/N flag.

    Roughly a quarter of values land outside the reference range, so the flag
    column carries real signal instead of being a wall of 'N'.
    """
    lo, hi = spec.ref_for(sex)
    roll = rng.random()
    if roll < 0.13:                       # low
        # Floor is proportional, not absolute: a WBC of 0.4 is not a low result,
        # it is incompatible with life, and such a value would make the whole
        # report obviously fake to anyone who reads lab work.
        value = rng.uniform(max(lo * 0.35, lo - spec.spread), lo * 0.995)
        return value, "L"
    if roll < 0.26:                       # high
        value = rng.uniform(hi * 1.005, min(hi * 1.8, hi + spec.spread))
        return value, "H"
    return rng.uniform(lo, hi), "N"


def _make_test(rng: random.Random, spec: TestSpec, sex: str) -> Test:
    unit = rng.choice(spec.units)
    raw, flag = _sample_value(rng, spec, sex)
    lo, hi = spec.ref_for(sex)

    value = _round(raw * unit.factor, unit.decimals)
    ref_low = _round(lo * unit.factor, unit.decimals)
    ref_high = _round(hi * unit.factor, unit.decimals)

    fmt = rng.choice(RANGE_FORMATS)
    ref_text = fmt.format(
        lo=f"{ref_low:.{unit.decimals}f}", hi=f"{ref_high:.{unit.decimals}f}"
    )

    return Test(
        test_name=rng.choice(spec.names),
        value=value,
        unit=unit.unit,
        panel=spec.panel,
        ref_low=ref_low,
        ref_high=ref_high,
        ref_text=ref_text,
        flag=flag,
        decimals=unit.decimals,
    )


def sample_report(rng: random.Random, index: int) -> Report:
    """Build one complete report with ground truth true by construction."""
    sex = rng.choice(("M", "F"))
    collected = date(2025, 1, 1) + timedelta(days=rng.randrange(365))
    reported = collected + timedelta(days=rng.choice((0, 0, 1)))

    first_names = MALE_NAMES if sex == "M" else FEMALE_NAMES
    patient = Patient(
        name=f"{rng.choice(first_names)} {rng.choice(LAST_NAMES)}",
        age=rng.randint(18, 82),
        sex=sex,
        patient_id=f"{rng.choice(('P', 'PT', 'REG'))}-{rng.randrange(100000, 999999)}",
    )
    meta = ReportMeta(
        lab_name=rng.choice(LAB_NAMES),
        collection_date=collected.strftime("%d/%m/%Y"),
        report_date=reported.strftime("%d/%m/%Y"),
    )

    tests = [_make_test(rng, spec, sex) for spec in CBC_TESTS]
    return Report(patient=patient, report=meta, tests=tests)


def sample_reports(n: int, seed: int = 0) -> list[Report]:
    """Deterministic for a given (n, seed). A sweep depends on this."""
    rng = random.Random(seed)
    return [sample_report(rng, i) for i in range(n)]
