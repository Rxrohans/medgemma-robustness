"""Emit paired PNG and ground-truth JSON.

    python -m generator.generate --n 5 --out data/synthetic/clean

The JSON beside each image is ground truth *by construction*: it is the same
Report object the template rendered, so it cannot disagree with the page. That
is the entire reason for generating data rather than labelling it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render import Renderer, available_templates
from .sampler import sample_reports


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic lab reports.")
    ap.add_argument("--n", type=int, default=5, help="number of reports")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/clean"))
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed keeps a benchmark reproducible")
    ap.add_argument("--scale", type=int, default=2,
                    help="device scale factor; 2 is ~192 DPI at A4")
    ap.add_argument("--template", default=None,
                    help="single template name; default cycles all available")
    args = ap.parse_args()

    templates = [args.template] if args.template else available_templates()
    if not templates:
        raise SystemExit(f"No templates found in {Path(__file__).parent / 'templates'}")

    reports = sample_reports(args.n, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    with Renderer(scale=args.scale) as renderer:
        for i, report in enumerate(reports):
            stem = f"{i:04d}"
            template = templates[i % len(templates)]

            renderer.render(report, template, args.out / f"{stem}.png")

            payload = report.to_dict()
            payload["_meta"] = {"template": template, "seed": args.seed, "index": i}
            (args.out / f"{stem}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    print(f"Wrote {len(reports)} report(s) to {args.out}")
    print(f"Templates used: {', '.join(templates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
