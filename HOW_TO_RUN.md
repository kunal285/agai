# How to Run the Project — PS-01 AI Dentistry

This guide provides step-by-step instructions to set up the environment, configure OpenAI Agentic AI credentials, and run the **PS-01 Deep Learning + Agentic AI Dental X-ray Screening System**.

---

## 1. Quick Start: Launch the 12-Section Streamlit Dashboard

Run from PowerShell or terminal in the project root:

```powershell
python -m streamlit run app.py
```

Once running, access the dashboard in your browser at:
`http://localhost:8501`

### Exploring the 12 PS-01 Dashboard Sections:
Use the sidebar navigation to toggle between:
1. **Screening Pipeline (Sections 1–9):**
   - **1. Dashboard Overview:** Architecture and academic disclaimer.
   - **2. Upload Dental X-ray:** Upload custom OPG or select from verified DENTEX test scans.
   - **3. Image Quality Analysis:** Agent 1 blur, contrast, and dimension metrics.
   - **4. YOLO Detection:** Agent 2 Part A bounding box overlays.
   - **5. CNN Classification:** Agent 2 Part B EfficientNet-B0 differential probabilities.
   - **6. Grad-CAM Explainability:** Agent 2 Part C activation heatmaps.
   - **7. Agentic AI Workflow:** 4-Agent execution timeline and OpenAI tool calling telemetry.
   - **8. Human Review:** Agent 3 clinical triage alert box and disagreement warnings.
   - **9. Screening Reports:** Agent 4 structured JSON report export and AI synthesis.
2. **Research Experiments (Sections 10–11):**
   - **10. The 7 Research Experiments:** Full comparative charts and tables for Exp 1 through Exp 7.
   - **11. Model Performance:** Summary metrics table, confusion matrices, and threshold sensitivity curves.
3. **System Settings (Section 12):**
   - **12. System Settings:** Model selector (`gpt-6.1-sol`, `gpt-6.2-astra`, `gpt-6-luna`), seed locking, hardware parameters, and privacy rules.

---

## 2. Environment Setup & Configuration

### Step 1: Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy the template `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Edit `.env`:
```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-6.1-sol
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=2
OPENAI_ENABLED=true
```

*(If `OPENAI_API_KEY` is not provided, the system automatically runs in 100% offline local inference mode with the four agents operating via deterministic rules).*

---

## 3. Run Automated Tests

To execute the complete 36-test automated verification suite:

```powershell
python -m pytest -q tests
```

Tests cover:
- Agent 1 Image Quality heuristics and rejection gate
- Agent 2 Detection, crop extraction, classification, and Grad-CAM
- Agent 3 Clinical Validation triage, threshold checks, and disagreement flagging
- Agent 4 Reporting Agent structured JSON schema validation
- Four-Agent Orchestrator tool calling with all 8 PS-01 named tools
- Offline inference fallback and OpenAI error recovery
- Privacy guardrail and PHI scrubbing
- Full regression across all existing pipelines
