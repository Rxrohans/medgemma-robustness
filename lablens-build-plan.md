# LabLens — Build Plan

**A robustness study of MedGemma 1.5 4B on medical lab report extraction.**

Zero budget. Zero proprietary data. Zero clinician dependency.

---

## What you are actually building

Three things, in order of portfolio value:

1. **A synthetic lab report generator** that produces realistic report images paired with perfect ground-truth JSON.
2. **An evaluation harness** that measures extraction accuracy field-by-field and reports where the model breaks.
3. **A working demo** that turns a lab report photo into structured FHIR data.

The headline result you are chasing: *"MedGemma 1.5 extraction accuracy holds until X, then collapses."* Nobody publishes that. Everybody publishes demos.

**The one-line pitch for your README:** "I independently measured MedGemma 1.5 4B's lab-report extraction on contamination-free synthetic data, across a controlled degradation sweep, against a base-Gemma-3 and a classical-OCR baseline."

### Get the target numbers right — this matters

Google's model card publishes these for PDF→JSON lab test extraction:

| Dataset | Gemma 3 4B | MedGemma 1.5 4B | Public? |
|---|---|---|---|
| EHR Dataset 2 | 84.0 Macro F1 | **91.0** | ❌ Google-internal |
| EHR Dataset 3 | 61.0 | 71.0 | ❌ Google-internal |
| **Mendeley Clinical Lab Reports** | *not published* | **85.0 Macro / 83.0 Micro** | ✅ free download |

Three things follow:

1. **The widely-quoted 91 is on a dataset you cannot access.** Do not frame your project as reproducing it. The only number you can check is **85 Macro F1 on Mendeley**.
2. **Google publishes no Gemma 3 4B baseline on Mendeley.** So your Phase 4 baseline #2 produces a genuinely new number rather than a reproduction. That is a real contribution — lead with it.
3. The 84→91 gap that everyone cites as proof medical pretraining works is measured on an unreleased dataset. Whether that gap survives on public data is an open question, and it is *your* question.

---

## Phase 0 — Setup (do this today, ~1 hour)

Some of these have waiting periods. Start the clocks now.

### 0.1 Hugging Face account — DO THIS FIRST

Free accounts can host ZeroGPU Spaces only if the **account is more than 30 days old** and the email is verified. That is your free demo hosting. Create the account today even though you won't deploy for six weeks.

1. Sign up at huggingface.co
2. Verify your email immediately
3. Go to the MedGemma model page (`google/medgemma-1.5-4b-it`) and accept the Health AI Developer Foundations terms. The model is gated; you cannot download weights until you accept.
4. Settings → Access Tokens → create a **read** token. Save it somewhere.

### 0.2 Kaggle account

1. Sign up at kaggle.com
2. Settings → **Phone verification**. This is mandatory. Without it the GPU accelerator dropdown stays locked.
3. Confirm you can see the accelerator options: GPU P100, GPU T4 x2, TPU.
4. Add your HF token: Notebook → Add-ons → Secrets → name it `HF_TOKEN`.

### 0.3 GitHub repo

Create it now, even empty. Commit from day one so the history shows the work.

```
lablens/
├── README.md
├── requirements.txt
├── generator/          # Phase 2
│   ├── schema.py
│   ├── sampler.py
│   ├── templates/
│   ├── render.py
│   └── degrade.py
├── extraction/         # Phase 3
│   ├── model.py
│   ├── prompts.py
│   └── run_batch.py
├── evaluation/         # Phase 4
│   ├── matching.py
│   ├── metrics.py
│   └── baselines.py
├── notebooks/          # Kaggle notebooks, exported
├── results/            # JSON + plots, committed
└── app/                # Phase 8, Gradio
```

### 0.4 LOINC account (Phase 7 — no rush)

Register free at loinc.org. **There is no approval queue** — create the account, accept the terms, download the table immediately. Organization affiliation is requested but is demographic data only; it gates nothing and is not verified. "Independent Researcher" is a fine answer.

Because there is no waiting period, this is *not* a Phase 0 task. Do it the day you start Phase 7.

---

## Phase 1 — Get the model running (½ day)

**Goal:** one image in, text out, on free hardware. Nothing more.

Open a Kaggle notebook, Accelerator → **GPU T4 x2** (preferred). The P100 works but has no FP16 tensor cores, so it is meaningfully slower on this workload. Use P100 only if T4s are unavailable.

```python
!pip install -q -U transformers accelerate bitsandbytes

from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

### The gotcha that will cost you an evening

Google's sample code uses `torch.bfloat16`. **Neither the T4 (Turing) nor the P100 (Pascal) supports bf16 in hardware.** Copy that line and you get a crash or a silent, miserable slowdown.

Use `float16`:

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-1.5-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,            # NOT bfloat16 on T4/P100
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

Two notes on that snippet:

- `torch_dtype=` is deprecated in current transformers; the argument is now `dtype=`. Google's sample code still shows the old name.
- You need **transformers ≥ 4.50** (the version that added Gemma 3 support). Kaggle's preinstalled version may be older or newer than the pinned torch expects — check and pin explicitly.

If you hit OOM, drop to 4-bit (puts the 4B model at roughly 5–7GB):

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto"
)
```

**Run one inference on any lab report image you can find** (screenshot one of your own, or grab a sample from the Mendeley set in Phase 6).

### 1.1 The resolution ceiling — resolve this before anything else

**MedGemma encodes every image at 896×896, into 256 tokens.** A full A4 lab report rendered at 300 DPI is 2550×3300; squashed to 896×896, the body text is close to illegible. This single fact threatens two parts of the plan:

- **"Render at 300 DPI as your clean baseline" is an illusion.** The model never sees those pixels. If your clean-baseline F1 comes out far below 85, suspect the resize before you blame the model.
- **The `scale` degradation axis — nominally your most important one — will produce a flat curve** from 1.0 down to roughly 0.35, then fall off a cliff the moment the long edge crosses under 896px. You would be plotting the behaviour of a resize function, not the robustness of a model.

**Mitigation:** Gemma 3 ships a *pan and scan* mode that tiles a large image into crops instead of squashing it. Turn it on:

```python
inputs = processor(
    text=prompt, images=img, return_tensors="pt",
    do_pan_and_scan=True,           # tiles instead of squashing
)
```

It costs more tokens and more time per image, so measure that cost here in Phase 1, not in Phase 5 when your quota is committed.

**Run this experiment now, on ~5 images.** It is 30 minutes and it determines the shape of the whole project:

| Condition | What you learn |
|---|---|
| 896-squash, full page | The naive baseline |
| Pan-and-scan, full page | Does tiling recover the small text? |
| 896-squash, cropped to the results table only | Is the problem resolution, or the model? |
| Seconds/image for each | Your real Phase 5 budget |

Then, on every plot you produce later, **report effective post-resize pixels alongside the nominal scale parameter.** Handled openly, this stops being a bug and becomes a finding: *a 4B VLM's document ceiling is set by its vision encoder, not its language model.* That is a better result than a clean curve.

### Record these numbers now — they go in your README

- Model load time
- Peak VRAM (`torch.cuda.max_memory_allocated() / 1e9`)
- Seconds per image at fp16 and at 4-bit
- Seconds per image with and without pan-and-scan
- Whether output quality differs between the two

That fp16-vs-4bit comparison is a free extra result. Most people never check whether quantization hurt them.

**Deliverable:** committed notebook, one working inference, timing table, and a go/no-go decision on pan-and-scan.

---

## Phase 2 — Synthetic data generator (1½ weeks, the biggest chunk)

This is the heart of the project. Build it in this order.

### 2.1 Define the schema first

Everything downstream depends on this. Write it before you write any rendering code.

```python
# generator/schema.py
REPORT_SCHEMA = {
    "patient": {"name": str, "age": int, "sex": str, "patient_id": str},
    "report": {"lab_name": str, "collection_date": str, "report_date": str},
    "tests": [{
        "test_name": str,
        "value": float,          # or str for qualitative
        "unit": str,
        "ref_low": float,
        "ref_high": float,
        "ref_text": str,         # some ranges are text, e.g. "Negative"
        "flag": str,             # H / L / N / None
        "panel": str,            # CBC, LFT, KFT, Lipid, Thyroid
    }]
}
```

Design note: include a `ref_text` field. Real reports mix numeric ranges with text ranges, and if your schema can't represent that, your eval is measuring the wrong thing.

### 2.2 Value sampler

Build realistic panels. You do not need clinical validity for an extraction test, but realism costs almost nothing and makes the demo credible.

Panels to cover: **CBC** (Hemoglobin, WBC, Platelets, RBC, MCV, MCH, MCHC), **LFT** (Bilirubin total/direct, SGOT/AST, SGPT/ALT, ALP, Albumin), **KFT** (Urea, Creatinine, Uric acid, Sodium, Potassium), **Lipid** (Total cholesterol, HDL, LDL, Triglycerides), **Thyroid** (TSH, T3, T4).

Deliberately build in the messiness that makes this hard:

- **Synonym pairs**: SGPT vs ALT, SGOT vs AST. Same test, different name. Real labs disagree.
- **Unit variants**: creatinine in mg/dL vs µmol/L, hemoglobin in g/dL vs g/L
- **Reference range formats**: `12.0 - 15.0`, `12.0–15.0` (en dash), `< 200`, `Up to 40`, `Negative`
- **Qualitative results**: "Negative", "Not Detected", "Trace"
- Sample some values inside range, some outside, so `flag` gets exercised

### 2.3 HTML templates

Use **Jinja2**. Build **6–8 visually distinct templates**. This is where diversity comes from, so don't cheap out.

Vary across templates:
- Single column vs two column
- Bordered tables vs zebra striping vs whitespace-only
- Serif vs sans-serif, different sizes
- Logo top-left vs centered vs none
- Header/footer block position and density
- Tests grouped by panel with subheadings, vs one flat table
- Some with a second page (tests overflow). Note that Google states MedGemma's multimodal capability was evaluated on **single-image tasks only** — multi-page was never tested. Keep these templates, feed pages as separate images, and report multi-page as a known-hard condition rather than treating a poor score as a bug.
- One template with a bilingual header, mimicking the Mendeley set's English/Arabic mix

### 2.4 Render to image — use Playwright, not WeasyPrint

```
HTML → (Playwright / headless Chromium) → PNG
```

```bash
pip install jinja2 playwright
playwright install chromium
```

**Why not WeasyPrint.** On Windows it needs Pango and Cairo, which means installing MSYS2 and pacman-installing GTK; then `pdf2image` needs a separate poppler binary. That is a multi-hour setup for a step you will run thousands of times, and it behaves differently on Windows than on Kaggle. Playwright is two commands, identical on both platforms, renders straight to PNG with no PDF hop, and has far better CSS support — which matters directly, because Phase 2.3's whole value is visually distinct templates.

```python
from playwright.sync_api import sync_playwright

def render(html: str, out_png: str, width=1240, scale=2):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": width, "height": 1754},
                        device_scale_factor=scale)   # scale=2 → ~200 DPI
        pg.set_content(html, wait_until="networkidle")
        pg.screenshot(path=out_png, full_page=True)
        b.close()
```

Launch the browser **once** and reuse it across all 300 reports — per-report launch is the difference between a 30-second run and a 10-minute one.

Render high-res as your clean baseline, but see Phase 1.1: anything above ~1200px on the long edge is thrown away by the model unless pan-and-scan is on. Keep the pristine version regardless — degradation is a separate step, and you may want the full-res originals later.

### 2.5 Degradation pipeline — this is what enables the headline result

Each axis must be **independently controllable with a numeric parameter**. That is what lets you sweep it later.

```python
# generator/degrade.py
def degrade(img, rotation=0, blur=0, jpeg_q=100, scale=1.0,
            perspective=0.0, shadow=0.0, noise=0.0):
    ...
```

Axes to implement, in priority order:

1. **Resolution / downscale** (`scale` 1.0 → 0.15). Most important. Simulates a bad phone photo.
2. **JPEG quality** (100 → 10). Compression artifacts, extremely common in WhatsApp-forwarded reports.
3. **Rotation** (0° → 15°). Scanner skew.
4. **Gaussian blur** (0 → 5px). Out-of-focus camera.
5. **Perspective warp** (0 → 0.3). Photo taken at an angle.
6. **Shadow / illumination gradient** (0 → 1.0). Hand shadow across the page.
7. **Gaussian noise** (0 → 1.0). Low-light sensor noise.

OpenCV and PIL handle all of these. Nothing exotic.

### 2.6 Emit paired output

```
data/synthetic/
├── clean/
│   ├── 0001.png
│   ├── 0001.json        ← ground truth, by construction
│   └── ...
└── degraded/
    └── rotation_05deg/
        ├── 0001.png
        └── 0001.json    ← same GT, image differs
```

**Generate 300 clean reports** for the main benchmark. For sweeps, reuse a fixed 100-report subset across every degradation level so differences are attributable to the degradation and not to sampling.

**Deliverable:** `python -m generator.generate --n 300 --out data/synthetic/clean` works end to end.

---

## Phase 3 — Extraction pipeline (3–4 days)

### 3.1 Prompt design

MedGemma is more prompt-sensitive than base Gemma 3. Google says so in the model card. Budget real time here.

Put the JSON schema **in the prompt**, demand strict JSON, forbid prose:

```python
PROMPT = """Extract all laboratory test results from this report as JSON.

Output ONLY valid JSON matching this schema, with no explanation,
no markdown fences, and no additional text:

{
  "patient": {"name": "", "age": null, "sex": ""},
  "tests": [
    {"test_name": "", "value": null, "unit": "",
     "ref_low": null, "ref_high": null, "ref_text": "", "flag": ""}
  ]
}

Rules:
- Copy test names exactly as printed on the report.
- If a value is qualitative (e.g. "Negative"), put it in "value" as a string.
- If a reference range is text rather than numeric, use "ref_text".
- Do not infer or invent any value that is not printed on the report.
"""
```

That last rule matters. You will measure hallucination as its own metric.

**Version your prompts.** Keep `prompts.py` with `PROMPT_V1`, `PROMPT_V2`, and record which prompt produced which result. A prompt-iteration table in your README is genuine evidence of methodology.

### 3.2 Robust parsing

The model will sometimes wrap output in fences or add a sentence. Handle it:

```python
def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None, "no_json_found"
    try:
        return json.loads(text[start:end+1]), None
    except json.JSONDecodeError as e:
        return None, f"parse_error: {e}"
```

**Log parse failures as a metric.** "JSON validity rate: 96.3%" is a real finding.

### 3.3 Batch runner with caching

You have 30 GPU hours a week. Do not waste them re-running finished work.

```python
# Cache by (image_hash, model_id, prompt_version)
# Skip anything already in results/raw/
```

Write results to disk as you go. Kaggle sessions die at 9–12 hours.

**Deliverable:** raw model outputs for all 300 clean reports, cached and committed.

---

## Phase 4 — Evaluation harness (4–5 days)

**This is the part that separates your project from a demo.** Spend real time on it.

### 4.1 Matching logic

Comparing predicted tests to ground-truth tests is not trivial. Decide and document these rules:

- **Test name matching**: normalize case and whitespace, then apply a synonym table (SGPT≡ALT). Report results both with and without the synonym table — the gap is itself a finding about whether the model normalizes terminology.
- **Value matching**: exact for integers; for floats use a relative tolerance (e.g. within 0.1%) to absorb float formatting.
- **Unit matching**: normalize (`mg/dl` ≡ `mg/dL`), and consider a unit-equivalence table separately from exact match.
- **Alignment**: match predicted tests to GT tests greedily on normalized name. Unmatched GT = miss. Unmatched prediction = hallucination.

### 4.2 Metrics to report

| Metric | Why it matters |
|---|---|
| Field-level precision / recall / F1 | The core number, comparable to Google's 91 |
| Macro F1 (per-document, averaged) | Matches Google's reported metric |
| Micro F1 (across all fields) | Also reported by Google |
| Per-field-type F1 | Which field is hardest? Usually reference ranges |
| **Hallucination rate** | Tests output that aren't on the page. Critical for medical |
| JSON validity rate | Did it produce parseable output at all |
| Document-perfect rate | % of reports extracted with zero errors |

Hallucination rate is your standout metric. In a medical context, inventing a test result is categorically worse than missing one, and almost no extraction paper separates the two.

### 4.3 Baselines — non-negotiable

Run all three on identical data:

1. **MedGemma 1.5 4B** (`google/medgemma-1.5-4b-it`)
2. **Gemma 3 4B** (`google/gemma-3-4b-it`) — same prompt, same everything
3. **Tesseract OCR + regex** — the dumb classical baseline

Baseline 2 is your real experiment. Google reports medical pretraining buying 91 vs 84 Macro F1 — **but only on EHR Dataset 2, which is internal and unreleased.** No Gemma 3 baseline is published on any public lab-report dataset, Mendeley included. So this comparison is not a reproduction; it is a new measurement, and it is the strongest single thing in the project. **Test it yourself.** If you find no meaningful gap on your data, that's a more interesting result than confirming the number, and you should say so plainly.

Baseline 3 establishes whether an LLM was needed at all. Sometimes it wasn't, and knowing that is a mark of maturity.

**Deliverable:** `results/main_benchmark.json` plus a markdown results table.

---

## Phase 5 — The robustness sweep (3–4 days)

The headline.

For each of the 7 degradation axes, take the fixed 100-report subset, run 6–8 levels, and plot F1 against the parameter.

```
For rotation:  0°, 2°, 5°, 8°, 10°, 15°
For scale:     1.0, 0.75, 0.5, 0.35, 0.25, 0.15
For jpeg_q:    100, 80, 60, 40, 25, 10
...
```

Sweep **one axis at a time**, holding others clean. Then do one combined "realistic phone photo" condition (moderate everything at once) to see if degradations compound worse than they add.

**GPU budget check — recalculated.** 7 axes × 6 levels × 100 reports = 4,200 inferences. The 4 s/inference figure is optimistic: a 4B VLM emitting 300–600 tokens of JSON runs **8–15 s** on a T4, and pan-and-scan pushes that higher. Realistic total: **10–18 hours**. That fits inside 30 h/week but consumes most of it, so:

- Run the main benchmark in week 1 and the sweep in week 2 — do not attempt both in one week.
- If you enable pan-and-scan, cut to **5 axes × 6 levels** (drop noise and shadow, the two least interesting) or drop to 60 reports per level.
- Time a single level end-to-end *first*, then multiply. Do not trust this estimate — trust your measurement.

Plot with matplotlib. Seven curves. Find and state the knee point on each.

**On the `scale` axis specifically:** label the x-axis with **effective pixels reaching the model**, not just the nominal scale factor. Per Phase 1.1, a squashed-to-896 pipeline will show a flat line then a cliff, and that cliff is the preprocessor, not the model. Say so explicitly in the caption. A reader who spots an artifact you did not flag stops trusting every other plot on the page.

Also run the sweep for Gemma 3 4B on at least the two most interesting axes, so you can say whether medical pretraining makes the model *more robust* or just more accurate on clean input. That's a genuinely novel question.

**Deliverable:** 7 plots in `results/plots/`, plus a paragraph naming each breaking point.

---

## Phase 6 — Real-world check (3 days)

### Getting the data

**Mendeley Clinical Laboratory Test Reports**, DOI `10.17632/bygfmk4rx9.2`. Search "Mendeley Clinical Laboratory Test Reports" and download from data.mendeley.com. Free, no credentialing, no application.

260 reports from 24 labs in Egypt, **CC BY 4.0** — free download, no credentialing, no application, Mendeley account only. Bodies in English, headers and footers in English and Arabic. Mixed acquisition: pdf-to-image, scanner-scanned, and mobile-scanned. That mix is exactly what you want. Attribute it properly in your README; CC BY requires it.

### Labeling

Hand-label **40 reports**. Tedious but purely mechanical — you are transcribing printed numbers, not making clinical judgments. Budget 4–6 hours. Build a tiny labeling helper: show the image, pre-fill with the model's output, you correct it. That cuts the time in half.

**Do not let the model's output bias you.** Label 10 reports blind first, compare your blind labels to your corrected labels, and report the discrepancy. That's your own annotation quality check.

### The mandatory caveat

Put this in your README: this dataset appears in Google's own published evaluation of MedGemma 1.5 — they report **85.0 Macro F1 / 83.0 Micro F1** on it. Numbers on it are a sanity check for real-world formats, not an independent measurement. Your synthetic results are the independent ones.

Do report your Mendeley number next to Google's 85, though. If you land far off it, that tells you something about your matching rules or your prompt before it tells you anything about the model — treat a large gap as a bug in your harness until proven otherwise.

Being explicit about this is a feature. It shows you understand contamination, which is the single most common blind spot in ML portfolio projects.

---

## Phase 7 — Normalization and FHIR (4–5 days)

This layer is deterministic code, not model output. The contrast is deliberate and worth pointing out in your writeup: use the LLM where it's needed, use ordinary code where it's better.

1. **LOINC mapping.** Download the LOINC table (free account from Phase 0). Map extracted test names to LOINC codes. Fuzzy-match with a hand-curated override table for the common Indian lab naming conventions. Report coverage: "82% of extracted tests mapped to LOINC."
2. **Unit conversion.** Canonical unit per LOINC code, conversion factors, flag anything unconvertible.
3. **FHIR Observation output.** Emit standards-compliant `Observation` resources. Validate with the free HL7 FHIR validator.
4. **Multi-report trending.** Given several reports for one patient, produce a time series per analyte. This is the actual user-facing value.

FHIR output is what makes this look professional rather than like a class assignment. The docs lean heavily on FHIR and it signals you know the domain's standards.

---

## Phase 8 — Demo and writeup (4–5 days)

### Gradio app on Hugging Face Spaces

Free accounts get **2 ZeroGPU Spaces**. Gradio SDK only (not Streamlit, not Docker). Visitors use their own daily GPU quota, so it costs you nothing.

```python
import spaces

@spaces.GPU(duration=60)
def extract(image):
    ...
```

Note: ZeroGPU hardware is modern, so you *can* use bfloat16 there, unlike on Kaggle. Keep the dtype configurable.

App flow: upload report → extracted table → normalized/LOINC view → FHIR JSON download.

### README structure

1. What this is, in two sentences
2. **The results table and the degradation plots, near the top.** Do not bury them.
3. Method: synthetic generator, why synthetic (contamination), degradation axes
4. Evaluation: matching rules, metrics, baselines
5. Findings, including the negative and surprising ones
6. Limitations, stated plainly
7. **Safety statement**: research and engineering demo, not for clinical use, no diagnostic claims, outputs require independent verification

Write limitations honestly. "I could not validate clinical correctness because I had no clinician access; this project measures extraction fidelity only, where ground truth is mechanical" is a *strong* sentence. It shows you know exactly what your evidence supports.

---

## Timeline

| Phase | Time | Output |
|---|---|---|
| 0. Setup | 1 hour | Accounts, repo, clocks started |
| 1. Model running | ½ day | Notebook + timing table |
| 2. Generator | 2–2½ weeks | 300 labeled synthetic reports |
| 3. Extraction | 3–4 days | Cached raw outputs |
| 4. Eval harness | 4–5 days | Benchmark table, 3 baselines |
| 5. Robustness sweep | 3–4 days | 7 plots, breaking points |
| 6. Real-world check | 3 days | Mendeley results + caveat |
| 7. FHIR + LOINC | 4–5 days | Standards-compliant output |
| 8. Demo + writeup | 4–5 days | Live Space, README |

**Realistically 8–10 weeks part-time.** Six is achievable only if Phase 2 goes perfectly first time; six to eight distinct templates plus a seven-axis degradation pipeline usually doesn't. Plan for eight to ten and treat finishing early as a win.

Note also that Phase 5 is gated on Kaggle's weekly quota reset, not on your own effort — the sweep cannot be compressed by working harder.

### If you need to cut scope

Drop Phase 7 (FHIR/LOINC) first. Phases 2, 4, and 5 are the project. A generator, a rigorous eval, and a robustness study with no FHIR layer is still a strong portfolio piece. A pretty demo with no evaluation is not.

---

## Things that will go wrong

- **bfloat16 on T4/P100.** Google's sample code really does use `torch.bfloat16`, and neither card supports it in hardware. Use float16. Also: the argument is now `dtype=`, not `torch_dtype=`.
- **The 896×896 resolution ceiling.** The biggest technical risk in the project. See Phase 1.1 — settle it before Phase 5, not during.
- **Kaggle session dies mid-run.** Sessions cap at 9 hours. Cache every result to disk immediately. Use Save & Run All (Commit) for background execution.
- **Prompt sensitivity.** MedGemma is more prompt-sensitive than Gemma 3. If results look terrible, iterate on the prompt before concluding the model is bad.
- **Playwright browser reuse.** Launching Chromium per report is ~20× slower than launching once and reusing the instance.
- **Quota burn.** 30 h/week, and the sweep alone is 10–18 h. Debug on 5 reports. Only run the full set when the pipeline is verified.
- **No local GPU.** Every inference runs on Kaggle, so you cannot debug model code offline. Keep the generator and the eval harness runnable on CPU locally, so only the extraction step needs the cloud — that separation is worth designing for from day one.
- **Scope creep into diagnosis.** Resist adding "and it tells you what's wrong." That breaks the prohibited-use policy, needs a clinician you don't have, and weakens the project.

---

## Start now

1. Create the Hugging Face account and verify email (30-day clock — the only genuinely urgent item).
2. Accept the MedGemma terms on the model page.
3. Phone-verify Kaggle, then add `HF_TOKEN` as a notebook secret.
4. Create the GitHub repo and push.

LOINC is not on this list: it has no approval queue, so it costs nothing to defer to Phase 7.

Then Phase 1 tomorrow.
