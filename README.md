# MedGemma Robustness

**How far can you degrade a lab report photo before a 4B medical vision-language
model stops reading it correctly?**

An independent evaluation of MedGemma 1.5 4B on lab report extraction: a synthetic
benchmark with ground truth by construction, a field-level evaluation harness, and
a controlled sweep across seven image-degradation axes to find the failure
threshold on each.

> 🚧 **Status: in progress.** Phase 0 (setup) complete. See
> [build-plan.md](build-plan.md) for the full plan.

---

## Results

*Nothing here yet. When there is, the results table and the degradation plots go
at the top of this file — not buried at the bottom.*

## What Google publishes, and what this project measures

| Dataset | Gemma 3 4B | MedGemma 1.5 4B | Public? |
|---|---|---|---|
| EHR Dataset 2 | 84.0 Macro F1 | 91.0 | ❌ Google-internal |
| EHR Dataset 3 | 61.0 | 71.0 | ❌ Google-internal |
| Mendeley Clinical Lab Reports | *not published* | 85.0 Macro / 83.0 Micro | ✅ free |

The frequently-quoted **91 Macro F1 is on an unreleased internal dataset** and
cannot be independently checked. The only reproducible public number is **85 on
Mendeley** — and no Gemma 3 baseline is published on any public lab-report set.
This project supplies that missing baseline, plus a robustness sweep nobody has run.

## Method

1. **Synthetic generator** — realistic lab report images with perfect ground-truth
   JSON by construction. Synthetic because Mendeley appears in Google's own
   evaluation, so it cannot serve as an independent measurement.
2. **Degradation pipeline** — seven independently parameterised axes (downscale,
   JPEG quality, rotation, blur, perspective, shadow, noise).
3. **Evaluation harness** — field-level P/R/F1, macro and micro, plus
   hallucination rate, JSON validity rate, and document-perfect rate.
4. **Baselines** — MedGemma 1.5 4B vs Gemma 3 4B vs Tesseract+regex, identical data.

## Repository layout

```
generator/    synthetic report generation + degradation   (CPU, local)
extraction/   model loading, prompts, batch runner        (GPU, Kaggle)
evaluation/   matching rules, metrics, baselines          (CPU, local)
notebooks/    exported Kaggle notebooks
results/      committed metrics + plots
app/          Gradio demo (Hugging Face Space)
```

Only `extraction/` needs a GPU. Everything else runs locally on CPU.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

## Limitations

To be written honestly, and before the results are flattering. Known already:

- This project measures **extraction fidelity only** — whether printed values are
  transcribed correctly. It makes no clinical correctness claims, because no
  clinician was involved.
- Synthetic reports are realistic but not real; they cannot capture every
  artifact of genuine lab output.
- MedGemma encodes images at 896×896, so results are partly a property of the
  vision encoder's resolution ceiling rather than the language model.

## Safety

Research and engineering demo. **Not for clinical use.** No diagnostic claims are
made and none should be inferred. Any output requires independent verification
before it is relied upon for any purpose.

## Data attribution

Mendeley Clinical Laboratory Test Reports (DOI `10.17632/bygfmk4rx9.2`),
260 reports from 24 laboratories in Egypt, licensed CC BY 4.0.
