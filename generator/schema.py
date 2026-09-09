"""Ground-truth schema for synthetic lab reports.

Everything downstream depends on this module: the sampler produces these
objects, the templates render them, and the evaluation harness compares model
output against them. Define it before writing any rendering code.

Design note: `value` is deliberately `float | str` and `ref_text` exists
alongside `ref_low`/`ref_high`. Real reports mix numeric ranges ("12.0 - 15.0")
with text ones ("Negative", "Up to 40"). A schema that cannot represent that
makes the evaluation measure the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Flag = Literal["H", "L", "N", ""]
Panel = Literal["CBC", "LFT", "KFT", "Lipid", "Thyroid"]


@dataclass
class Patient:
    name: str
    age: int
    sex: str
    patient_id: str


@dataclass
class ReportMeta:
    lab_name: str
    collection_date: str
    report_date: str


@dataclass
class Test:
    test_name: str          # as printed — synonyms (SGPT vs ALT) are the point
    value: float | str      # str for qualitative results ("Negative", "Trace")
    unit: str
    panel: Panel
    ref_low: float | None = None
    ref_high: float | None = None
    ref_text: str = ""      # used when the range is not two numbers
    flag: Flag = ""


@dataclass
class Report:
    patient: Patient
    report: ReportMeta
    tests: list[Test] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to the ground-truth JSON written next to each image."""
        return asdict(self)


# The subset of fields the model is asked to produce, and that Phase 4 scores.
# Keep this in sync with the schema embedded in extraction/prompts.py.
SCORED_TEST_FIELDS = ("test_name", "value", "unit", "ref_low", "ref_high",
                      "ref_text", "flag")
SCORED_PATIENT_FIELDS = ("name", "age", "sex")
