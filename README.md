# MedGemma Robustness

**How far can you degrade a lab report photo before a 4B medical vision-language
model stops reading it correctly?**

An independent evaluation of MedGemma 1.5 4B on lab report extraction: a
synthetic benchmark with ground truth by construction, a field-level evaluation
harness, and a controlled sweep across seven image-degradation axes to find the
failure threshold on each.

> **Status: in progress.** Generator under construction.

## Results

Nothing here yet. When there is, the results table and the degradation plots go
at the top of this file, not buried at the bottom.

## What Google publishes, and what this measures

| Dataset | Gemma 3 4B | MedGemma 1.5 4B | Public? |
|---|---|---|---|
| EHR Dataset 2 | 84.0 Macro F1 | 91.0 | No, Google-internal |
| EHR Dataset 3 | 61.0 | 71.0 | No, Google-internal |
| Mendeley Clinical Laboratory Test Reports | not published | 85.0 Macro / 83.0 Micro | Yes, free |

The frequently quoted 91 Macro F1 is measured on an unreleased internal dataset
and cannot be independently checked. The only reproducible public number is 85
on Mendeley, and no Gemma 3 baseline is published on any public lab-report set.
This project supplies that missing baseline and adds a robustness sweep that has
not been run before.

## Method

1. **Synthetic generator.** Realistic lab report images with ground-truth JSON
   produced by construction rather than by labelling. Synthetic because the
   Mendeley set appears in Google's own evaluation of this model, so it cannot
   serve as an independent measurement.
2. **Degradation pipeline.** Seven independently parameterised axes: downscale,
   JPEG quality, rotation, blur, perspective, shadow, and sensor noise.
3. **Evaluation harness.** Field-level precision, recall and F1, macro and micro,
   plus hallucination rate, JSON validity rate, and document-perfect rate.
4. **Baselines.** MedGemma 1.5 4B against base Gemma 3 4B and against Tesseract
   with regex, all on identical data.

## Repository layout

```
generator/    synthetic report generation and degradation   (CPU, local)
extraction/   model loading, prompts, batch runner          (GPU, Kaggle)
evaluation/   matching rules, metrics, baselines            (CPU, local)
notebooks/    exported Kaggle notebooks
results/      committed metrics and plots
app/          Gradio demo
```

Only `extraction/` needs a GPU. Everything else runs locally on CPU, which keeps
development off the weekly GPU quota.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
playwright install chromium

python -m generator.generate --n 5 --out data/synthetic/clean
```

## Limitations

To be written honestly, and before the results are flattering. Known already:

- This measures **extraction fidelity only**, meaning whether printed values are
  transcribed correctly. It makes no clinical correctness claims, because no
  clinician was involved.
- Synthetic reports are realistic but not real, and cannot capture every artefact
  of genuine lab output.
- MedGemma encodes images at 896x896 into 256 tokens, so results are partly a
  property of the vision encoder's resolution ceiling rather than of the language
  model.

## Safety

Research and engineering demo. **Not for clinical use.** No diagnostic claims are
made and none should be inferred. Any output requires independent verification
before it is relied upon for any purpose.

## Licensing

Three different terms apply to three different things here, and they are not
interchangeable:

| What | Terms |
|---|---|
| This repository's code | [Apache License 2.0](LICENSE) |
| MedGemma 1.5 4B and Gemma 3 4B weights | [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms), gated, accepted on the model page |
| Mendeley Clinical Laboratory Test Reports | CC BY 4.0, attribution required |

The Apache-2.0 grant covers the generator, the evaluation harness and the app in
this repository. It grants no rights to the model weights, which remain under
Google's terms and must be obtained from Hugging Face directly.

Dataset citation: *Clinical Laboratory Test Reports*, DOI
[`10.17632/bygfmk4rx9.2`](https://doi.org/10.17632/bygfmk4rx9.2), 260 reports
from 24 laboratories in Egypt, CC BY 4.0.
