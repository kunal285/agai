"""Detection and diagnosis agent module."""
from dentistry.agents.detection_diagnosis_agent import DetectionDiagnosisAgent

# DentalDetectionAgent provides the standalone detector-only interface
class DentalDetectionAgent:
    name = "dental_detection"

    def __init__(self, detector, confidence_threshold: float = 0.075):
        self.detector = detector
        self.confidence_threshold = confidence_threshold

    def run(self, image, confidence_threshold: float | None = None):
        diag_agent = DetectionDiagnosisAgent(self.detector, classifier=None, confidence_threshold=confidence_threshold or self.confidence_threshold)
        return diag_agent.run(image)

__all__ = ["DetectionDiagnosisAgent", "DentalDetectionAgent"]
