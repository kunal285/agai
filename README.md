# Agentic AI-Assisted Dental X-ray Screening

Educational research prototype for panoramic dental X-ray quality screening, abnormal-region localization, crop-level classification, Grad-CAM visualization and a four-agent reporting workflow. It is **not a clinical diagnostic or treatment system**.

## Current status

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for verified environment and implementation/test status. No DENTEX data or trained checkpoints are bundled. The app deliberately refuses to fabricate predictions when checkpoints are missing.

## Architecture

`upload → Image Quality Agent → YOLO detector → region crops → EfficientNet/ResNet classifier → Grad-CAM → Validation Agent → Reporting Agent → JSON/dashboard`

The four software agents are deterministic Python components, not a chatbot. The Validation Agent flags low-confidence or missing model outputs for human review. Detector and classifier adapters require real checkpoint files.

## Repository layout

- `dentistry/dataset.py` — inventory, COCO parsing, validation, YOLO conversion and annotated crop extraction.
- `dentistry/models.py` — checkpoint-backed YOLO and torchvision classifier adapters, coordinate-preserving crops.
- `dentistry/training.py` — YOLO and EfficientNet/ResNet training entry points.
- `dentistry/gradcam.py` — Grad-CAM heatmap/overlay generation.
- `dentistry/agents/` — four required agents.
- `dentistry/orchestrator.py` — ordered end-to-end workflow and JSON reports.
- `dentistry/metrics.py` — classification metrics and documented IoU matching helpers.
- `scripts/` — dataset inspection, conversion, group-aware crop split, training and YAML generation.
- `app.py` — Streamlit dashboard.
- `tests/` — synthetic software fixtures only; never evidence of model accuracy.
- `docs/` — methodology and four-student/eight-week plan.

## Requirements and setup

Python 3.10–3.12 is recommended for broad ML wheel compatibility. The inspected runtime was Python 3.13 with CPU-only PyTorch; Ultralytics and Streamlit were not installed. A GPU is optional but strongly recommended for training.

### Linux/macOS

```bash
cd ai_dentistry
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[models,app,test]'
```

### Windows PowerShell

```powershell
cd ai_dentistry
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e "[models,app,test]"
```

If your platform cannot install the optional ML dependencies, install base/test dependencies with `pip install -e '.[test]'`; dataset validation, quality checks, agents and tests that do not require the optional packages remain usable. Use the PyTorch selector at https://pytorch.org/get-started/locally/ for the right CPU/CUDA wheel.

## Dataset acquisition and inspection

Obtain DENTEX from its official challenge/data page and comply with its access and use terms. Place the extracted files under `data/DENTEX/` (or provide another root). Dataset layout and annotation schema are not assumed.

```bash
python scripts/inspect_dataset.py --root data/DENTEX --output artifacts/dataset_inventory.json
```

Review `annotation_summaries`, category names/IDs, missing image associations, invalid boxes and official splits before preparing anything. The converter supports COCO-style JSON; if the downloaded DENTEX package differs, adapt the parser only after inspecting its actual schema. Do not assume the four diagnostic labels are present in every annotation file.

```bash
python scripts/prepare_dataset.py path/to/verified_annotations.json \
  --image-root path/to/images \
  --yolo-out data/processed/yolo/labels \
  --crops-out artifacts/crops
python scripts/split_crop_manifest.py artifacts/crops/manifest.csv \
  --output artifacts/crops/split_manifest.csv
```

**Important:** `prepare_dataset.py` creates labels and crop files; it does not rearrange the original images into Ultralytics' `images/{train,val,test}` layout. Build those directories using verified split manifests (or official splits) and corresponding labels before training. Keep source images grouped across splits. Prefer official split assignments when supplied by DENTEX.

Create a detector YAML only after the class mapping and image/label directories are confirmed:

```bash
python scripts/create_yolo_yaml.py --dataset-root data/processed/yolo \
  --classes caries deep_caries periapical_lesion impacted_tooth \
  --output data/processed/dentex.yaml
```

The class names above are an example only: use the actual category mapping in your selected detector task. The default four classifier labels are the diagnostic classes specified by the project brief, but they must be confirmed against the real annotation files.

## Training

Check free disk/RAM/GPU resources first. Start with small pilot runs, inspect label overlays, and only then run full experiments.

```bash
python scripts/train_detector.py --data data/processed/dentex.yaml --epochs 50 --imgsz 640 --batch 8
python scripts/train_classifier.py --manifest artifacts/crops/split_manifest.csv --epochs 20 --batch 16
```

YOLO saves its `best.pt` under the training output directory; copy or configure the resulting path as `models/detector/best.pt`. The classifier saves `models/classifier/best.pt` by default, including its actual training classes and metadata. The detector training code uses a pretrained YOLO initialization by default; use `--resume` only when a compatible last checkpoint exists.

## Evaluate and compare

`dentistry.metrics` exposes classification metrics and a one-to-one greedy IoU matcher. For final research results, evaluate the trained detector with Ultralytics validation on the held-out test split and export the actual metrics. Evaluate the classifier with predictions from the locked checkpoint on the held-out classifier split, then call `classification_metrics`. Implement and document the exact detector-only versus detector+CNN matching/label aggregation rule before comparing end-to-end macro-F1. Do not tune thresholds on test data.

Experiment outputs should include dataset manifest hash, config, seed, package versions, checkpoint hashes, per-class metrics, confusion matrix, threshold sweep, runtime, and error examples. Pending experiments must remain marked pending; this repository does not ship invented metrics.

## Run the dashboard

```bash
streamlit run app.py
```

Upload a PNG/JPEG/TIFF/BMP panoramic X-ray. The dashboard will only run model inference when the required real checkpoints and packages exist. Configure paths and thresholds in `configs/default.yaml`. JSON reports are written to `artifacts/reports/`; Grad-CAM outputs are stored in `artifacts/gradcam/` when supported.

## Tests

```bash
pytest
```

The tests use synthetic arrays/images to test software behavior (annotation conversion, invalid inputs, model-loading failures, quality checks, agent order, report generation, metrics and Grad-CAM). They do **not** measure real-world model performance.

## Limitations

- DENTEX is not bundled; schema and category IDs must be verified against the actual download.
- No trained checkpoints are bundled. Inference will raise a clear error until training succeeds.
- The default image quality thresholds are heuristic and require validation.
- Classifier confidence is an uncalibrated score unless separately calibrated.
- Grad-CAM indicates model sensitivity, not proof of pathology.
- Dataset/source-level grouping must be maintained to prevent leakage.
- Research-only; qualified clinicians must interpret images. Never use this prototype to make treatment decisions.

## Academic deliverables

See `docs/METHODOLOGY.md` for the research protocol and `docs/TEAM_AND_8_WEEKS.md` for team roles, schedule and viva prompts. Expand the literature review with 10–12 verified papers and cite primary sources/DOIs only after checking the exact bibliographic details.

### Classifier held-out evaluation

After training and ensuring the crop manifest contains a distinct `test` split, run:

```bash
python scripts/evaluate_classifier.py --manifest artifacts/crops/split_manifest.csv \
  --checkpoint models/classifier/best.pt --output artifacts/metrics/classifier_test.json
```

This script refuses to evaluate if no test rows exist. It writes per-class metrics, confusion matrix, and individual predictions. Keep the test split untouched until the model and thresholds are finalized.
# agai
