"""
ConcreteGuard — Prediction Engine
==================================

Loads the trained model and provides inference with label mapping,
uncertainty handling, and visual-priority proxy.
"""

import json
import os
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
CLASS_NAMES = ["no_crack", "crack"]

# Default values (overridden by metadata files when available)
DEFAULT_THRESHOLD = 0.50
DEFAULT_UNCERTAINTY_MARGIN = 0.10


class CrackPredictor:
    """Loads the trained model and provides prediction with metadata."""

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: Path to model_artifacts/ directory containing:
                - best_model.keras (or best_model.h5)
                - threshold.json
                - preprocessing.json
                - class_names.json
                - model_metadata.json
        """
        self.model_dir = model_dir
        self.model = None
        self.threshold = DEFAULT_THRESHOLD
        self.uncertainty_margin = DEFAULT_UNCERTAINTY_MARGIN
        self.class_names = CLASS_NAMES
        self.preprocessing = {"rescale": 1.0 / 255}

        self._load_metadata()
        self._load_model()

    def _load_metadata(self):
        """Load threshold, preprocessing, and class name metadata."""
        # Threshold
        threshold_path = os.path.join(self.model_dir, "threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                data = json.load(f)
                self.threshold = data.get("threshold", DEFAULT_THRESHOLD)
                self.uncertainty_margin = data.get("uncertainty_margin", DEFAULT_UNCERTAINTY_MARGIN)
            print(f"  Threshold: {self.threshold:.2f} (+/- {self.uncertainty_margin:.2f})")

        # Preprocessing
        preproc_path = os.path.join(self.model_dir, "preprocessing.json")
        if os.path.exists(preproc_path):
            with open(preproc_path) as f:
                self.preprocessing = json.load(f)

        # Class names
        class_path = os.path.join(self.model_dir, "class_names.json")
        if os.path.exists(class_path):
            with open(class_path) as f:
                self.class_names = json.load(f)

    def _load_model(self):
        """Load the Keras model."""
        import tensorflow as tf

        for ext in [".keras", ".h5"]:
            model_path = os.path.join(self.model_dir, f"best_model{ext}")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"  Model loaded from {model_path}")
                return

        print(f"  [WARN] No model file found in {self.model_dir}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess a PIL Image for model input."""
        # Ensure RGB
        if image.mode != "RGB":
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            else:
                image = image.convert("RGB")

        # Resize
        image = image.resize(IMG_SIZE, Image.BILINEAR)

        # To array and scale
        arr = np.array(image, dtype=np.float32)
        rescale = self.preprocessing.get("rescale", 1.0 / 255)
        arr = arr * rescale

        return arr

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single image.
        
        Returns:
            {
                "label": str,
                "probability": float,
                "threshold": float,
                "visual_priority": str,
                "review_recommendation": str,
                "limitations": str,
            }
        """
        if self.model is None:
            return {
                "label": "model_unavailable",
                "probability": 0.0,
                "threshold": self.threshold,
                "visual_priority": "manual_review",
                "review_recommendation": "Model not loaded. Manual review required.",
                "limitations": "This is not a structural safety assessment.",
            }

        arr = self.preprocess(image)
        prob = float(self.model.predict(arr[np.newaxis], verbose=0).flatten()[0])

        # Three-way label assignment with adaptive uncertainty bounds
        # Ensure lower bound is strictly positive so negative images can be classified as no_visible_crack
        effective_margin = min(self.uncertainty_margin, self.threshold * 0.4)
        lower = max(0.01, self.threshold - effective_margin)
        upper = min(0.99, self.threshold + effective_margin)

        if prob < lower:
            label = "no_visible_crack"
            visual_priority = "low"
            recommendation = (
                "No visible crack detected in this image. "
                "This does not rule out hidden or structural damage."
            )
        elif prob > upper:
            label = "potential_crack"
            visual_priority = "high" if prob > 0.85 else "medium"
            recommendation = "Potential crack detected. Submit the image for expert inspection."
        else:
            label = "uncertain_manual_review"
            visual_priority = "manual_review"
            recommendation = (
                "Image quality or model confidence is insufficient. "
                "Manual review is recommended."
            )

        return {
            "label": label,
            "probability": round(prob, 4),
            "threshold": self.threshold,
            "visual_priority": visual_priority,
            "review_recommendation": recommendation,
            "limitations": "This is not a structural safety assessment. "
                           "It detects visible crack-like image patterns only.",
        }

    def predict_with_gradcam(self, image: Image.Image) -> dict:
        try:
            from gradcam import generate_gradcam, gradcam_to_base64
        except ImportError:
            from .gradcam import generate_gradcam, gradcam_to_base64

        result = self.predict(image)

        if self.model is not None:
            arr = self.preprocess(image)
            heatmap = generate_gradcam(self.model, arr[np.newaxis])
            result["gradcam_base64"] = gradcam_to_base64(arr, heatmap)
            result["gradcam_heatmap"] = heatmap
        else:
            result["gradcam_base64"] = None
            result["gradcam_heatmap"] = None

        return result
