# Four-person team plan (8 weeks)

| Student | Primary ownership | Shared integration responsibilities |
|---|---|---|
| 1 | Dataset acquisition, audit, EDA, preprocessing, manifests | Verify data provenance and leakage prevention |
| 2 | YOLO conversion, training, detection evaluation | Explain detector failures and metrics |
| 3 | CNN crops, transfer learning, classifier evaluation, Grad-CAM | Explain class imbalance and XAI limitations |
| 4 | Four agents, orchestrator, Streamlit, integration tests | Demo end-to-end flow and failure routing |

## Weekly milestones
1. Literature review, DENTEX access, environment and data audit.
2. Annotation mapping, EDA, fixed group-aware splits.
3. YOLO baseline and validation error review.
4. CNN crop classifier and validation checkpoint selection.
5. Detector-vs-combined evaluation and Grad-CAM examples.
6. Agent rules, report schema, workflow tests.
7. Streamlit integration, confidence threshold and preprocessing experiments.
8. Locked test evaluation, research report, presentation and viva practice.

## Viva preparation prompts
- Why are the detector and classifier separate?
- How were DENTEX category IDs verified?
- What prevents crop-level train/test leakage?
- What is the difference between precision, recall, mAP and macro-F1?
- Why is a high model score not automatically a calibrated probability?
- What does Grad-CAM show, and what can it not establish?
- How does the Validation Agent route uncertain or failed inference?
- Which failure cases and dataset limitations remain?
