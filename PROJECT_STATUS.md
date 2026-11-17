# Project status and verification log

Last inspected: 2026-10-25

## Environment inspection

- Workspace: fresh repository created at `/mnt/data/ai_dentistry`; no pre-existing source files were found in `/mnt/data`, `/home/oai/share` or `/workspace`.
- Specification: `01_Simplified_AI_Dentistry_150_Marks.docx` found in the user's Library and materialized to `/mnt/data/01_Simplified_AI_Dentistry_150_Marks.docx`.
- Python: 3.13.5.
- PyTorch: 2.10.0+cpu; CUDA unavailable.
- Available at inspection: OpenCV, NumPy, Pandas, scikit-learn, torchvision, Pillow, pytest.
- Not installed at inspection: Streamlit, Ultralytics.
- DENTEX dataset: not found in the workspace. No real images, labels or dataset statistics are claimed.
- Trained model checkpoints: none generated.

## Feature ledger

| Area | Status | Evidence / note |
|---|---|---|
| Configuration and package layout | IMPLEMENTED | `configs/default.yaml`, `pyproject.toml` |
| Dataset inventory / COCO parser / validation | TESTED (missing-data path) | `scripts/inspect_dataset.py` correctly reports dataset root absent; parser/conversion unit tests pass |
| YOLO label conversion / crop extraction | IMPLEMENTED | `coco_to_yolo`, `extract_crops` |
| Group-aware crop split utility | IMPLEMENTED | `scripts/split_crop_manifest.py`; needs at least 3 source image IDs |
| YOLO training/inference adapter | IMPLEMENTED, NOT TRAINED | Ultralytics dependency/checkpoint required |
| CNN training/inference adapter | IMPLEMENTED, NOT TRAINED | Torchvision checkpoint required |
| Grad-CAM | IMPLEMENTED | `dentistry/gradcam.py` |
| Four agents and orchestration | IMPLEMENTED | `dentistry/agents`, `dentistry/orchestrator.py` |
| JSON reporting and review routing | IMPLEMENTED | Reports contain actual model output only |
| Streamlit dashboard source | IMPLEMENTED, NOT RUN | Streamlit absent at environment inspection; source compiles but UI runtime not verified |
| Metrics / IoU matching helpers | TESTED | Unit tests cover classification metrics and IoU matching |
| Automated tests | TESTED | `pytest -q`: 14 passed; synthetic fixtures only |
| Real dataset EDA / real training / final evaluation | PENDING | DENTEX absent; no checkpoints |
| Research references / complete literature review | PENDING | Must be filled with verified sources, not invented citations |

## Required honesty rules

Synthetic test fixtures are only software tests. No real-world accuracy, mAP, clinical benefit or model superiority may be claimed until trained models are evaluated on a held-out real DENTEX test set.

## Execution verification

- `python -m compileall -q dentistry scripts app.py`: passed.
- `pytest -q`: **14 passed**.
- `python scripts/inspect_dataset.py --root data/DENTEX ...`: ran and correctly reported that the dataset root does not exist.
- YOLO training: NOT RUN (dataset absent, Ultralytics not installed, CPU-only environment).
- CNN training: NOT RUN (dataset absent; no split manifest).
- Dashboard runtime: NOT RUN (Streamlit not installed).
- Real model metrics: NONE measured.
