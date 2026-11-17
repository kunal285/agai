# Methodology and research protocol

## Research question
Does detector-plus-CNN confirmation improve abnormality classification compared with detector-only output, while maintaining interactive inference speed? The comparison is empirical; no improvement is assumed.

## Dataset protocol
Use the official DENTEX data and its licensing/terms. Inspect every annotation JSON before assigning labels. DENTEX has hierarchical tasks, so only use the fully annotated abnormality-diagnosis subset and verify category IDs from its metadata. Keep source image identifiers grouped across splits to avoid crop-level leakage. Official split definitions should be preferred when available and appropriate. Never tune on the held-out test set.

## Models
A YOLO-family detector localizes annotated regions. A torchvision EfficientNet-B0 classifier is trained on padded crops extracted from diagnostic annotations. Both model families are optional runtime dependencies. The detector is trained with the actual verified label schema; the classifier label vocabulary is derived from observed training labels and saved with the checkpoint.

## Metrics
Report detector precision, recall, mAP@0.5 and mAP@0.5:0.95 from the detector validation/evaluation outputs. Report classifier per-class precision, recall, F1, macro-F1, confusion matrix and classification report. For detector-vs-combined comparison, define a matching rule before test evaluation; report IoU threshold, one-to-one matching, confidence thresholds, and treatment of missed/extra detections. Save all results with dataset manifest hashes, configuration, model checkpoint IDs, seed and software versions.

## Preprocessing experiment
Compare raw grayscale/RGB input with a fixed, documented contrast normalization variant. Keep all other training settings constant. Select settings on validation only; evaluate test once after locking the experiment.

## Explainability
Grad-CAM is generated from the classifier's last convolutional feature block. It is a model-sensitivity visualization, not proof of pathology and not a validated clinical explanation. Include examples of correct, incorrect, and uncertain predictions after real training.

## Limitations and safety
This prototype is for academic research only. Dataset labels may not represent all clinical conditions, image quality varies, confidence is not calibrated probability, and neither the model nor agent workflow is clinically validated. A qualified dental professional must interpret imaging; this software must not recommend treatment.
