# PHASE1_REPORT.md — ConcreteGuard Evaluation Report

> **Status:** Template — actual metrics will be filled after Colab training completes.

---

## 1. Overall Test Set Metrics

| Metric | Value |
|---|---|
| Accuracy | *pending training* |
| Precision | *pending training* |
| Recall | *pending training* |
| F1-score | *pending training* |
| Total test samples | — |
| False positives | — |
| False negatives | — |

Confusion matrix: see `evaluation_outputs/confusion_matrix.png`

---

## 2. Hard-Negative False-Positive Rate — Separate from Overall Accuracy

| Set | Total | False Positives | FPR |
|---|---:|---:|---:|
| All hard negatives | — | — | — |
| Stains | — | — | — |
| Joints | — | — | — |
| Shadows | — | — | — |

> **Note:** This FPR is computed only on hard-negative images (all ground truth `no_crack`). It is reported separately and is not merged into overall accuracy.

---

## 3. Per-Category Analysis

### Stains
Stains create discoloration that can produce dark, crack-like visual features. The model's stain FPR measures how well it distinguishes discoloration from actual cracks.

### Construction Joints
Joints are linear features that structurally resemble cracks. The model must learn that these are intentional construction elements.

### Shadows
Shadow boundaries create high-contrast edges similar to crack edges. Shadow FPR is particularly important because shadows are ubiquitous in outdoor inspection photos.

---

## 4. Cost Assumptions

```
C_FN = 10   (relative cost of missed visible crack)
C_FP = 1    (relative cost of unnecessary inspection)
```

### Interpretation
- **A missed crack** may delay inspection and create greater risk of undetected deterioration
- **An unnecessary inspection** consumes time and resources but does not create safety risk
- The 10:1 ratio reflects that missing a potentially significant crack is much more costly than flagging a clean surface

These are **relative planning costs**, not financial or engineering claims. They guide threshold selection to favor recall over precision.

---

## 5. Threshold Justification

- Threshold selected on **validation set only** using cost-minimization
- The chosen threshold is the lowest-cost operating point under the documented cost assumptions
- An uncertainty margin (±0.10) routes borderline cases to manual review
- The threshold was **frozen** before evaluating on the test set

See `evaluation_outputs/threshold_cost_curve.png` for the cost vs. threshold curve.

---

## 6. Correct Grad-CAM Example

*See `evaluation_outputs/correct_case_gradcam.png`*

The Grad-CAM heatmap for a correctly identified crack shows activation concentrated along the crack line, confirming the model attends to the relevant structural feature rather than background texture.

---

## 7. Error Case — Wrong Prediction with Grad-CAM Evidence

*See `evaluation_outputs/error_case.json`, `error_case_original.png`, `error_case_gradcam.png`*

### Details

| Field | Value |
|---|---|
| Ground truth | *pending* |
| Predicted | *pending* |
| Probability | *pending* |
| Error type | *pending* |

### Explanation

> *The written explanation will be generated from the actual error case after training. It will include what the model focused on (from Grad-CAM), why the visual feature likely confused the model, and why human review is the correct action.*

### Takeaway
This error demonstrates that the model can be misled by visual patterns that share local texture characteristics with cracks. The appropriate response is to route uncertain or erroneous cases for expert review rather than treating any model output as a structural conclusion.

---

## 8. Limitations

1. **Dataset bias:** Training data comes from a limited set of structures and lighting conditions. Performance on novel environments is not guaranteed.

2. **Patch leakage:** The Mendeley dataset does not provide source-image grouping. Patches from the same high-resolution source may appear in multiple splits, potentially inflating metrics.

3. **Binary simplification:** All cracks are treated equally regardless of width, depth, or structural significance.

4. **No structural assessment:** The system detects visible crack-like patterns only. It cannot assess structural integrity, crack depth, or safety implications.

5. **Hard-negative coverage:** The hard-negative set covers stains, joints, and shadows. Other confusing patterns (e.g., scratches, paint lines, rough texture) may not be well represented.

6. **Image quality dependency:** Low-resolution, blurry, or poorly lit images reduce model reliability. The quality check module provides warnings but cannot guarantee correct results on marginal images.
