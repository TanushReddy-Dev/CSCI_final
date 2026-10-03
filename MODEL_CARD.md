# MODEL_CARD.md — ConcreteGuard

## Model Overview

| Property | Value |
|---|---|
| Model name | ConcreteGuard MobileNetV2 |
| Architecture | MobileNetV2 (ImageNet) + GlobalAvgPool + Dropout(0.3) + Dense(1, sigmoid) |
| Task | Binary classification: crack vs. no visible crack |
| Input | 224 × 224 × 3 RGB image, rescaled to [0, 1] |
| Output | Sigmoid probability (0 = no crack, 1 = crack) |
| Framework | TensorFlow / Keras |
| Training | Google Colab with T4 GPU |

---

## Intended Use

### Primary use case
Visual screening aid that helps maintenance teams prioritize images of concrete surfaces for expert inspection.

### Intended users
- Maintenance teams performing visual surveys
- Building inspectors using the tool as a first-pass filter
- Researchers evaluating crack detection approaches

### Out-of-scope uses

⚠️ **This model must NOT be used for:**
- Structural safety assessment
- Engineering decisions about building integrity
- Automated safety clearance without human review
- Determining structural failure risk
- Replacing professional engineering inspection

---

## Training Details

### Two-phase training

1. **Phase 1 — Head training (5 epochs):**
   - MobileNetV2 backbone frozen
   - Only classification head trained
   - Learning rate: 1e-3 (Adam)

2. **Phase 2 — Fine-tuning (7 epochs):**
   - Last 30 backbone layers unfrozen
   - Learning rate: 1e-5 (Adam)

### Data augmentation (training only)
- ±15° rotation
- Brightness/contrast adjustment (0.8–1.2×)
- 10% zoom
- 5% translation
- Horizontal flip
- No augmentation applied to validation or test sets

### Loss and metrics
- Loss: Binary cross-entropy
- Tracked: Accuracy
- Saved: Best model by validation loss

---

## Threshold Selection

The decision threshold is selected on the **validation set only** using cost-minimization:

```
C_FN = 10  (relative cost of missed crack)
C_FP = 1   (relative cost of unnecessary inspection)

expected_cost = C_FN × FN + C_FP × FP
normalized_cost = expected_cost / N_samples
```

The threshold that minimizes normalized cost is selected. An uncertainty margin (default ±0.10) creates a three-way classification:

| Condition | Label |
|---|---|
| probability < threshold − margin | `no_visible_crack` |
| probability within margin | `uncertain_manual_review` |
| probability > threshold + margin | `potential_crack` |

### Rationale
- A missed crack may delay inspection and create greater risk
- An unnecessary inspection consumes time and resources
- The system prioritizes recall while monitoring hard-negative FPR
- The threshold is the lowest-cost operating point under documented cost assumptions

---

## Evaluation Metrics

*Populated after training. See `evaluation_outputs/metrics.json` for actual values.*

### Standard test set metrics

| Metric | Value |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-score | — |
| False positives | — |
| False negatives | — |

### Hard-negative false-positive rate (separate from overall accuracy)

| Set | Total | False Positives | FPR |
|---|---:|---:|---:|
| All hard negatives | — | — | — |
| Stains | — | — | — |
| Joints | — | — | — |
| Shadows | — | — | — |

---

## Known Failure Modes

1. **Shadow edges:** High-contrast shadow boundaries can resemble elongated cracks
2. **Construction joints:** Linear joints may be confused with structural cracks
3. **Surface stains:** Dark discoloration patterns may trigger false positives
4. **Thin/low-contrast cracks:** Hairline cracks on light surfaces may be missed
5. **Distant/blurry captures:** Low image quality reduces model confidence

---

## Human Review Recommendation

All model outputs should be treated as **screening suggestions**, not diagnostic conclusions:

- **`potential_crack`** → Submit for expert inspection
- **`no_visible_crack`** → Does not rule out hidden or structural damage
- **`uncertain_manual_review`** → Confidence insufficient, manual review required
- **`image_quality_insufficient`** → Image too blurry/dark/small for reliable analysis

---

## Ethical Considerations

- The model learns from a specific dataset of concrete surfaces and may not generalize to all construction materials, lighting conditions, or geographic regions
- False negatives (missed cracks) carry higher real-world risk than false positives
- The system should supplement, never replace, professional structural inspection
- No claims about building safety should be derived from model outputs alone
