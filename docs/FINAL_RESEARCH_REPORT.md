# University Research Report (150 Marks)
## Agentic AI-Assisted Dental X-Ray Screening for Abnormal Tooth Detection and Diagnosis
**Course:** Senior Capstone / Deep Learning Project (8-Week University Deliverable)  
**Dataset:** DENTEX Challenge 2023 (`data/archive/training_data/training_data/quadrant-enumeration-disease/`)  
**Status:** Reproducible Research Prototype & Empirical Benchmark  

> [!CAUTION]
> **ACADEMIC & EDUCATIONAL DISCLAIMER**  
> This system is an educational research prototype developed strictly for academic investigation. It is **NOT** a clinical diagnostic, prognostic, or treatment-planning medical device. Panoramic radiographs must always be inspected and interpreted by a licensed dental practitioner.

---

## 1. Executive Summary & Research Problem

Manual inspection of panoramic radiographs (OPGs) is mentally taxing, prone to subjective oversight, and time-intensive. Dental radiographs feature multi-scale pathology—ranging from tiny interproximal enamel demineralization (caries) to extensive bone crypt radiolucencies (periapical lesions) and anatomical malpositions (impacted teeth).

### Primary Research Question (RQ1)
> **"Does a two-stage detector-plus-CNN confirmation pipeline improve macro-F1 over an end-to-end single-stage YOLO detector alone, while remaining fast enough for interactive clinical screening (<150 ms)?"**

### Empirical Findings Summary
| Metric | Detector-Only (YOLOv8n) | Two-Stage (YOLOv8n + EfficientNet-B0) | Delta ($\Delta$) | Clinical/Engineering Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Macro-F1** | 0.1582 | 0.1407 | -0.0175 | Detector captures holistic spatial context; isolated small crops suffer from loss of alveolar boundary context. |
| **Precision** | 0.4052 | 0.3739 | -0.0313 | Confirmation stage rejected several true positives with ambiguous boundaries. |
| **Recall** | 0.1053 | 0.0989 | -0.0064 | Bounded by Stage 1 region proposal recall ceiling. |
| **Latency per Image** | **103.8 ms** | **332.7 ms** | +228.9 ms | Detector-only is fully interactive (9.6 FPS); two-stage introduces moderate overhead (~3 FPS on CPU). |

---

## 2. Dataset & Zero-Leakage Experimental Protocol

### 2.1 Dataset Provenance & Integrity
- **Source:** DENTEX Challenge 2023 (Official Grand Challenge / Kaggle).
- **Sub-Task:** Fully annotated quadrant-enumeration-disease subset (`train_quadrant_enumeration_disease.json`).
- **Images:** 705 panoramic radiographs (`xrays/`), 0 corrupted, 0 hash duplicates.
- **Mean Native Resolution:** $2830.3 \times 1366.6$ px (Aspect ratio: $2.08:1$).
- **Annotations:** 3,529 bounding boxes mapped to four canonical classes:
  1. `caries` (Class ID 1 in DENTEX $\rightarrow$ Canonical ID 0): 2,096 instances (59.4%)
  2. `deep_caries` (Class ID 3 in DENTEX $\rightarrow$ Canonical ID 1): 584 instances (16.5%)
  3. `periapical_lesion` (Class ID 2 in DENTEX $\rightarrow$ Canonical ID 2): 208 instances (5.9%)
  4. `impacted_tooth` (Class ID 0 in DENTEX $\rightarrow$ Canonical ID 3): 641 instances (18.2%)

### 2.2 Leakage Prevention Strategy
A critical failure mode in medical imaging research is **crop-level patient leakage**—where crops from the same radiograph appear in both training and test splits, artificially inflating validation accuracy.
- **Strict Image-Level Multi-Label Stratified Split:**
  All 705 radiographs were partitioned strictly at the image level using iterative greedy multi-label stratification (`results/split_manifest.csv`):
  - **Train:** 493 radiographs (70.0%), 2,467 instances
  - **Validation:** 106 radiographs (15.0%), 531 instances
  - **Held-Out Test:** 106 radiographs (15.0%), 531 instances
- **Test Integrity:** The test set remained completely locked and untouched during all training and hyperparameter selection phases.

---

## 3. Deep Learning Architectures

Strict project constraint: Exactly **two** core deep learning architectures.

```
                    ┌───────────────────────────────┐
                    │  Raw Panoramic Radiograph     │
                    │      (2830 x 1366 px)         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Image Quality Agent      │
                    │   (Laplacian blur, Contrast)  │
                    └───────────────┬───────────────┘
                                    │ Acceptable
                                    ▼
                    ┌───────────────────────────────┐
                    │    Stage 1: YOLOv8n Detector  │
                    │ (640x640, Anchor-free head)   │
                    └───────────────┬───────────────┘
                                    │ Candidate BBoxes + Conf
                                    ▼
                    ┌───────────────────────────────┐
                    │   Context-Padded Cropping     │
                    │   (10% anatomical margin)     │
                    └───────────────┬───────────────┘
                                    │ 224x224 RGB Crops
                                    ▼
                    ┌───────────────────────────────┐
                    │  Stage 2: EfficientNet-B0 CNN │
                    │   (Transfer Learning + CAM)   │
                    └───────────────┬───────────────┘
                                    │ Logits & Activation Maps
                                    ▼
                    ┌───────────────────────────────┐
                    │    Validation & Report Agent  │
                    │ (Triage Low-Conf to Dentist)  │
                    └───────────────────────────────┘
```

1. **Stage 1 — Object Detector (Ultralytics YOLOv8n):**
   - Size: 3.01M parameters, 8.1 GFLOPs.
   - Purpose: Multi-scale anatomical localization of abnormal dental regions on full $640 \times 640$ letterboxed radiographs.
   - Loss Function: Complete IoU (CIoU) + Distribution Focal Loss (DFL) + Binary Cross-Entropy.
2. **Stage 2 — Crop Classifier (torchvision EfficientNet-B0):**
   - Size: 4.01M parameters, pretrained on ImageNet-1K.
   - Purpose: Fine-grained differential diagnosis on $224 \times 224$ crops extracted with 10% spatial context padding.
   - Loss Function: Cross-Entropy with inverse-frequency class weights $[0.42, 1.51, 4.24, 1.38]$ to counteract severe class imbalance.

---

## 4. Empirical Research Findings & Viva Answers

### RQ1: Detector-Only vs. Two-Stage Confirmation Pipeline
- **Hypothesis:** Stage 2 CNN would filter false positive detections and improve macro-F1.
- **Empirical Result:** Detector-only achieved **Macro-F1 = 0.1582** (Latency: **103.8 ms**); Two-Stage achieved **Macro-F1 = 0.1407** (Latency: **332.7 ms**).
- **Research Insight:** In dental radiograms, isolated tooth crops eliminate essential surrounding landmarks (e.g., alveolar crest level, mandibular nerve canal proximity, adjacent crown comparison). In Stage 2, without global contextual reference, borderline lesions can be misclassified, while latency triples.
- **Conclusion:** For interactive chairside screening, a well-calibrated single-stage detector is significantly faster and preserves global jaw topology.

### RQ2: Abnormality Difficulty & Class Imbalance Analysis
- **Impacted Tooth:** Easiest to detect ($P=0.332, R=0.419, \text{mAP@50}=0.292$) due to distinct macroscopic angulation and high-contrast radiopaque crown density.
- **Caries:** High recall ($R=0.576, \text{mAP@50}=0.256$) due to large sample representation (59.4% of dataset).
- **Deep Caries & Periapical Lesions:** Hardest to detect ($\text{mAP@50}=0.049$ and $0.005$). When a $2830 \times 1366$ radiograph is downsampled to $640 \times 640$, a $30 \times 30$ px apical radiolucency shrinks to under $7 \times 7$ px, losing subtle trabecular bone pattern contrast.

### RQ3: Preprocessing Impact — Raw vs. CLAHE Contrast Enhancement
- **Raw Radiograph:** **Macro-F1 = 0.1582**
- **CLAHE Enhancement (clipLimit=2.0, grid=8x8):** **Macro-F1 = 0.1083** ($\Delta = -0.0498$)
- **Clinical/Technical Explanation:** Contrast-Limited Adaptive Histogram Equalization amplifies high-frequency noise in the radiolucent maxillary sinuses and mandibular bone marrow, inducing detector false positives along normal anatomical trabeculae.

### RQ4: Explainability via Grad-CAM
- Grad-CAM heatmaps generated from EfficientNet-B0's final convolutional layer (`features[-1]`) reveal strong gradient localization over the occlusal/interproximal enamel junction for caries.
- **Failure Analysis:** In low-confidence predictions, activations diffuse into the surrounding periodontal ligament space, indicating uncertainty.

### RQ5: Agentic Workflow & Human-in-the-Loop Triage
The 4-agent deterministic pipeline successfully routes cases:
1. **Image Quality Agent:** Evaluates brightness, contrast, and Laplacian blur variance ($\sigma_{\text{Laplacian}}^2 < 10$). Rejects degraded/blurred scans before DL inference.
2. **Detection & Diagnosis Agent:** Executes batch inference and extracts anatomical bounding boxes.
3. **Validation Agent:** Flags predictions with confidence $< 0.35$ or ambiguous classes for mandatory human dentist review.
4. **Reporting Agent:** Generates structured tooth-wise screening manifests with unique screening IDs and disclaimer metadata.

---

## 5. Team Work Distribution & 8-Week Timeline

| Student | Primary Ownership | Deliverables & Code Artifacts |
| :--- | :--- | :--- |
| **Student 1 (Data & Preprocessing)** | Dataset Audit, EDA, Splitting, Manifests | `results/split_manifest.csv`, `results/dataset_report.json`, Zero-leakage verification test |
| **Student 2 (Object Detection)** | YOLOv8 Training, Validation, mAP Benchmarks | `main.py`, `models/best_model.pt`, `data/processed/yolo/dentex_yolo.yaml` |
| **Student 3 (CNN & Explainability)** | Padded Crop Pipeline, EfficientNet-B0, Grad-CAM | `scripts/extract_dentex_crops.py`, `models/classifier/best.pt`, `dentistry/gradcam.py` |
| **Student 4 (Agents & Dashboard)** | 4-Agent Orchestrator, Streamlit UI, Test Suite | `dentistry/orchestrator.py`, `app.py`, `tests/`, Pytest verification suite |

### 8-Week Milestone Schedule
- **Week 1:** Literature review of 12 dental AI papers; local environment setup and DENTEX data provenance audit.
- **Week 2:** Coordinate conversion (COCO to YOLO), EDA on aspect ratios, image-level stratified splitting.
- **Week 3:** YOLOv8n detector baseline training on CPU/GPU, loss monitoring, mAP@50 evaluation.
- **Week 4:** Padded crop extraction (3,529 crops), class weighting, EfficientNet-B0 transfer learning.
- **Week 5:** Grad-CAM implementation, hook registration, visual alignment analysis.
- **Week 6:** 4-Agent deterministic rule engine, review routing thresholds, unit test suite.
- **Week 7:** Streamlit interactive dashboard with real-time bounding box and Grad-CAM visualization.
- **Week 8:** Locked test set evaluation, empirical research report, viva preparation and defense.
