from scripts.calibrate_detector import evaluate


def test_calibration_counts_duplicates_and_wrong_classes(monkeypatch, tmp_path):
    box = [0, 0, 10, 10]
    predictions = [
        {"bbox": box, "detector_class": "caries", "detector_confidence": 0.9},
        {"bbox": box, "detector_class": "caries", "detector_confidence": 0.3},
        {"bbox": box, "detector_class": "impacted_tooth", "detector_confidence": 0.8},
    ]

    class Detector:
        def predict(self, image):
            return predictions

    monkeypatch.setattr("scripts.calibrate_detector.read_image", lambda path: None)
    images = [("x.png", [{"bbox": box, "label": "caries"}])]
    scores = evaluate(Detector(), images, tmp_path, [0.2, 0.5], ["caries", "impacted_tooth"])

    assert scores[0.2]["per_class"]["caries"]["fp"] == 1
    assert scores[0.2]["per_class"]["impacted_tooth"]["fp"] == 1
    assert scores[0.5]["per_class"]["caries"]["tp"] == 1
    assert scores[0.5]["macro_f1"] == 0.5
