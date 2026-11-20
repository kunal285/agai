# Agentic AI-Assisted Dental X-Ray Screening

> **DISCLAIMER:** This project is an **educational research prototype only**. It is **NOT** a clinical diagnostic, prognostic, or treatment-planning system. All machine-learning outputs and AI synthesis reports must be verified by a licensed dental practitioner.

---

## 1. Project Overview & Architecture

This system implements an advanced, production-grade **Multi-Agent AI Orchestration Architecture** combined with deep-learning computer vision models trained on the **DENTEX Challenge 2023** panoramic dental radiograph dataset.

### Core 7-Agent Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          Raw Panoramic Radiograph            │
                    │               (OPG Upload)                   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │        Agent 1: Master Orchestrator          │
                    │   (OpenAI Tool Calling / Local Fallback)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │        Agent 2: Image Quality Agent          │
                    │   (Laplacian Blur, Contrast, Dimensions)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      │ Passed Quality                          │ Failed Quality Gate
                      ▼                                         ▼
        ┌───────────────────────────┐             ┌───────────────────────────┐
        │ Agent 3: Detection Agent  │             │ Agent 6: Validation Agent │
        │   (YOLOv8n Full Canvas)   │             │   (Route: Dentist Review) │
        └─────────────┬─────────────┘             └─────────────┬─────────────┘
                      │ Proposed Bounding Boxes                 │
                      ▼                                         │
        ┌───────────────────────────┐                           │
        │ Agent 4: Classifier Agent │                           │
        │  (EfficientNet-B0 Crops)  │                           │
        └─────────────┬─────────────┘                           │
                      │ Diagnosis Confirmation                  │
                      ▼                                         │
        ┌───────────────────────────┐                           │
        │ Agent 5: Explainability   │                           │
        │   (Grad-CAM Saliency)     │                           │
        └─────────────┬─────────────┘                           │
                      │ Visual Heatmaps                         │
                      ▼                                         │
        ┌───────────────────────────┐                           │
        │ Agent 6: Validation Agent │◄──────────────────────────┘
        │   (Safety Triage Rules)   │
        └─────────────┬─────────────┘
                      │ Validated Screening Manifest
                      ▼
        ┌───────────────────────────┐
        │ Agent 7: Reporting Agent  │
        │ (Structured JSON + Synth) │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Streamlit Web Dashboard  │
        └───────────────────────────┘
```

---

## 2. Multi-Agent System Roles & Responsibilities

1. **Agent 1: Master Orchestrator (`dentistry/agents/orchestrator_agent.py`)**  
   Coordinates the complete multi-turn screening workflow using the official OpenAI Responses API and tool calling. Prevents infinite execution loops, tracks per-agent latencies, monitors token costs, and deterministically falls back to local execution if the API is offline.
2. **Agent 2: Image Quality Agent (`dentistry/agents/quality_agent.py`)**  
   Evaluates radiograph sharpness via Laplacian variance ($\sigma_{\text{Laplacian}}^2 < 15$), brightness bounds ($10 \le \mu \le 245$), contrast standard deviation ($\sigma \ge 10$), and minimum dimensions. Rejects corrupted or degraded scans before deep-learning inference.
3. **Agent 3: Dental Detection Agent (`dentistry/agents/detection_agent.py`)**  
   Executes the calibrated Ultralytics YOLOv8n detector (`models/best_model.pt`) on the full-canvas radiograph to propose candidate bounding boxes across the 4 canonical DENTEX categories: `caries`, `deep_caries`, `periapical_lesion`, `impacted_tooth`.
4. **Agent 4: Dental Classification Agent (`dentistry/agents/classification_agent.py`)**  
   Extracts 10% context-padded bounding-box crops ($224 \times 224$ px) and executes the compact torchvision EfficientNet-B0 classifier (`models/classifier/best.pt`). Compares predictions against detector proposals and flags model disagreements.
5. **Agent 5: Explainability Agent (`dentistry/agents/explainability_agent.py`)**  
   Computes gradient-weighted class activation mapping (Grad-CAM) registered to the classifier's final convolutional layer (`features[-1]`). Overlays model feature sensitivity heatmaps.
6. **Agent 6: Clinical Validation Agent (`dentistry/agents/validation_agent.py`)**  
   Enforces non-diagnostic clinical boundaries. Flags any finding with low confidence ($P_{\text{det}} < 0.25$, $P_{\text{cls}} < 0.65$), model disagreement, missing crops, or quality failures for mandatory qualified dentist review. Never declares an X-ray "healthy" or "normal".
7. **Agent 7: Reporting Agent (`dentistry/agents/reporting_agent.py`)**  
   Synthesizes all findings into a structured, audit-ready JSON report containing the analysis ID, image quality assessment, candidate detections, classifications, model agreement, Grad-CAM references, execution trace, and token telemetry.

---

## 3. Supported OpenAI Models & Tool Registry

### Configurable Model Identifiers
- **Primary Model:** `gpt-6.1-sol` (Balanced reasoning and orchestration speed)
- **Advanced Model:** `gpt-6.2-astra` (Deep medical contextual synthesis)
- **Budget Model:** `gpt-6-luna` (Low latency and minimal token cost)

### Real Agentic Tools (`dentistry/llm/tool_registry.py`)
- `validate_xray_image(image_ref)`
- `run_quality_check(image_ref)`
- `run_yolo_detection(image_ref, confidence_threshold)`
- `run_efficientnet_classification(region_id, crop_bbox_xyxy)`
- `generate_gradcam(region_id)`
- `compare_model_predictions(region_id, detector_class, ...)`
- `validate_detection_results(findings)`
- `request_dentist_review(reason, flagged_regions)`
- `generate_structured_report(analysis_id, synthesis_notes)`

---

## 4. Environment Setup & API Configuration

### Step 1: Copy Environment Template
Create your `.env` configuration file from the template:
```powershell
cp .env.example .env
```

### Step 2: Configure `.env`
Edit `.env` with your settings:
```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-6.1-sol
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=2
OPENAI_ENABLED=true
```

> [!IMPORTANT]
> **API Security & Privacy:**
> - `.env` is listed in `.gitignore` and is never committed to source control.
> - API keys are never exposed in logs, reports, or the Streamlit frontend.
> - **Patient Privacy Guardrail (`dentistry/safety/privacy.py`):** Raw dental radiographs and patient-identifying data (names, MRNs, dates) are **never** transmitted to the external LLM API. Only anonymized bounding box coordinates, class labels, and quality metrics are sent.

---

## 5. How to Run

### Run the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

#### Dashboard Capabilities:
- Live 7-Agent execution progress and per-agent latency telemetry.
- Interactive confidence sliders for detector and classifier.
- Bounding box overlay visualization directly on panoramic X-rays.
- Visual side-by-side comparison of candidate crops and Grad-CAM sensitivity heatmaps.
- Model disagreement callouts and mandatory dentist review triage alerts.
- Downloadable structured JSON clinical screening reports.
- Real-time token usage and estimated API cost monitor.

### Run Automated Tests
```powershell
python -m pytest tests
```
Runs the complete test suite including:
- Unit tests for all 7 agents
- Tool registry validation & safe dispatching
- Quality gate rejection on blurred/corrupted scans
- Model disagreement and human review routing
- PHI scrubbing and privacy verification
- Offline fallback and API error recovery
- Full regression across existing pipelines

---

## 6. Offline / Fallback Operation

If no OpenAI API key is supplied, or if network connectivity is absent, the system **automatically and gracefully falls back** to the local deterministic rule engine. All computer vision models (YOLOv8n, EfficientNet-B0, and Grad-CAM) run locally with zero cloud dependencies.
