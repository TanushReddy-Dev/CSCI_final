# DATASET_CARD.md — ConcreteGuard

## Primary Dataset

**Name:** Concrete Crack Images for Classification  
**Source:** Mendeley Data  
**URL:** https://data.mendeley.com/datasets/5y9wdsg2zt/2  
**License:** CC BY 4.0  
**Citation:** Özgenel, Ç.F. (2019). Concrete Crack Images for Classification. Mendeley Data, V2.

### Composition

| Property | Value |
|---|---|
| Total images | 40,000 |
| Positive (crack) | 20,000 |
| Negative (no crack) | 20,000 |
| Image size | 227 × 227 RGB |
| Source images | 458 high-resolution photographs |

### Known Limitations

1. **Patch leakage risk:** Images are 227×227 patches extracted from 458 high-resolution source images. If patches from the same source image appear in both training and test sets, evaluation metrics may be overly optimistic. The original dataset does not provide source-image grouping, so perfect source-level splitting is not possible without manual curation.

2. **Domain gap:** All images come from a limited set of structures and lighting conditions. Performance may degrade on surfaces with different concrete types, colors, lighting, or camera quality.

3. **Binary simplification:** Real-world crack severity varies. This dataset treats all cracks equally, regardless of width, depth, or structural significance.

---

## Hard-Negative Evaluation Set

Hard negatives are concrete surface images that contain **no cracks** but have visual features that may confuse the model.

### Categories

| Category | Description | Min Target Count |
|---|---|---|
| Stains | Water stains, discoloration, rust marks | 30 |
| Construction joints | Expansion joints, formwork lines | 30 |
| Shadows | Cast shadows creating high-contrast edges | 30 |

### Additional categories (if available)

- Scratches
- Paint lines
- Rough texture
- Holes/chips
- Low-quality/blurry captures

### Ground Truth

Every hard-negative image is ground truth `no_crack`. The purpose is to measure how often the model incorrectly flags these visual patterns as cracks.

### Manifest Format

```csv
image_path,category,ground_truth,source,lighting,capture_notes
hard_negatives/stain/img001.jpg,stain,no_crack,curated,natural,Water stain on exterior wall
```

---

## Data Split Strategy

### Preferred: Source-Level Splitting

```
Source group A → training (70%)
Source group B → validation (15%)
Source group C → test (15%)
```

### Fallback: Stratified Random Split

If source grouping is unavailable, use stratified 70/15/15 random split with fixed seed (42).

**Limitation:** Random splitting may allow patches from the same high-resolution source image to appear in multiple splits, inflating reported accuracy. This limitation must be disclosed in the Phase 1 report.

---

## External Test Data

Team-captured images kept completely separate from all training, validation, and threshold selection:

- Fresh crack photographs
- Clean concrete surfaces
- Stain/joint/shadow examples
- Intentionally blurry, dark, or distant captures

**These images must never be used for threshold selection or model tuning.**
