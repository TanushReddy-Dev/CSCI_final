# JIGNASA 2026 Master Build Plan
## Explainable Concrete Surface Crack Inspection System

> **Purpose:** Give this document to an implementation agent and have it build the complete project: dataset pipeline, Colab training, evaluation evidence, model export, and a polished frontend generated through Stitch MCP and connected to inference.

---

# 1. Product definition

## 1.1 Product name

Use a professional working name such as **ConcreteGuard** or **CrackSense**. The final name may be changed by the team.

## 1.2 Product statement

Build an explainable visual screening assistant that analyzes photographs of concrete surfaces and helps maintenance teams prioritize images that may require expert inspection.

The system must:

- Accept an uploaded concrete-surface image.
- Predict `crack` or `no_crack`.
- Show confidence/probability.
- Display Grad-CAM evidence.
- Provide a cautious visual-priority proxy.
- Warn about poor image quality or uncertainty.
- Explicitly test stains, construction joints, and shadows as hard negatives.
- Report hard-negative-only false-positive rate separately from overall accuracy.
- Show at least one model failure with evidence and an explanation.

## 1.3 Safety boundary

This is **not** a structural-safety detector. It must never claim:

- The building is safe.
- The building is unsafe.
- Structural integrity is confirmed.
- Structural failure is ruled out.
- No engineer is needed.

Use this language instead:

> Potential crack detected. Submit the image for expert inspection.

> No visible crack detected in this image. This does not rule out hidden or structural damage.

> Image quality or model confidence is insufficient. Manual review is recommended.

---

# 2. Agent operating instructions

Give the entire repository and this plan to the implementation agent.

## 2.1 Agent must inspect first

Before modifying anything:

1. Inspect the current repository tree.
2. Identify the existing frontend, backend, notebooks, and configuration.
3. Preserve working code.
4. Determine whether a Stitch MCP tool is actually available.
5. Determine whether the project already has an inference API or only a trained model.
6. Use existing conventions where possible.
7. Do not replace a working project with a new scaffold without checking it first.

## 2.2 Agent must work in phases

Implement in this order:

1. Product and repository setup.
2. Data and model-training contract.
3. Google Colab training code.
4. Evaluation and required evidence.
5. Model export and inference contract.
6. Stitch MCP frontend generation.
7. Frontend integration.
8. Testing and demo hardening.
9. Documentation and final handoff.

Do not start with visual polish before the model contract and evaluation artifacts exist.

## 2.3 Agent must keep an implementation log

Create `BUILD_LOG.md` and record:

- Date/time.
- What was changed.
- Commands run.
- Results.
- Known issues.
- Next action.

If a required external tool or connector is unavailable, record the limitation and use the documented fallback rather than silently pretending it was used.

---

# 3. Target architecture

## 3.1 Recommended architecture for the hackathon

```text
Google Colab
  ├── Download/prepare public dataset
  ├── Train MobileNetV2 transfer-learning model
  ├── Evaluate standard test set
  ├── Evaluate hard negatives
  ├── Select threshold by relative cost
  ├── Generate Grad-CAM artifacts
  └── Export model + metadata

Frontend generated with Stitch MCP
  ├── Upload image
  ├── Preview image
  ├── Call inference endpoint or local adapter
  ├── Show result card
  ├── Show confidence and visual priority
  ├── Show Grad-CAM overlay
  ├── Show image-quality warning
  ├── Show hard-negative examples/dashboard
  └── Show disclaimer and expert-review action
```

## 3.2 Integration modes

The agent must support one of these modes, in this order:

### Mode A — API-backed inference, preferred

- Colab exports the trained model.
- A lightweight Python inference server loads it.
- Frontend sends an image to `POST /predict`.
- Server returns prediction, confidence, Grad-CAM image, quality status, and review recommendation.

### Mode B — Precomputed demo mode, acceptable fallback

- Frontend includes a fixed set of demo images and precomputed results.
- Upload can be simulated by selecting demo images.
- The interface must clearly label this as demo mode if live inference is not connected.

### Mode C — Client-side model inference

Use only if the exported model is compatible with the chosen frontend runtime. Do not force this if it complicates Grad-CAM or creates unreliable browser performance.

The final implementation must clearly document which mode is active.

---

# 4. Repository deliverables

The agent must produce the following structure, adapting to the existing project if necessary:

```text
project-root/
├── README.md
├── BUILD_LOG.md
├── DATASET_CARD.md
├── MODEL_CARD.md
├── PHASE1_REPORT.md
├── requirements.txt
├── .env.example
├── frontend/
│   ├── README.md
│   ├── src/
│   ├── public/
│   └── ...
├── inference/
│   ├── app.py
│   ├── predictor.py
│   ├── gradcam.py
│   ├── quality.py
│   └── requirements.txt
├── colab/
│   ├── train_crack_model.py
│   ├── train_crack_model.ipynb  # optional generated notebook
│   └── README.md
├── model_artifacts/
│   ├── best_model.keras or best_model.h5
│   ├── class_names.json
│   ├── model_metadata.json
│   ├── threshold.json
│   └── preprocessing.json
├── evaluation_outputs/
│   ├── metrics.json
│   ├── metrics.csv
│   ├── confusion_matrix.png
│   ├── threshold_cost_curve.png
│   ├── hard_negative_results.csv
│   ├── hard_negative_summary.json
│   ├── error_case.json
│   ├── error_case_original.png
│   ├── error_case_gradcam.png
│   └── sample_predictions.csv
└── data_manifests/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── hard_negatives.csv
```

If a native backend already exists, integrate into it rather than adding a second server.

---

# 5. Data plan

## 5.1 Primary public data

Use SDNET2018 if available and practical. Otherwise use the Concrete Crack Images for Classification dataset from Mendeley Data.

Record the official URL, citation, license, class counts, and limitations in `DATASET_CARD.md`.

For the Mendeley dataset, document:

- 40,000 RGB images.
- 20,000 positive crack images.
- 20,000 negative images.
- 227 × 227 image size.
- Images derived from 458 high-resolution source images.
- CC BY 4.0 attribution.
- Risk of overly optimistic results if patches from the same source are randomly split.

## 5.2 Hard-negative data

Create a separate evaluation set with at least:

- 30 stains.
- 30 construction joints.
- 30 shadows.

If possible, add:

- Scratches.
- Paint lines.
- Rough texture.
- Holes/chips.
- Low-quality images.

Every hard-negative image is ground truth `no_crack`, but its category must be preserved.

Required manifest columns:

```text
image_path,category,ground_truth,source,lighting,capture_notes
```

## 5.3 External test data

Keep new team-captured images separate from training and validation:

- Visible cracks.
- Clean concrete.
- Stains.
- Joints.
- Shadows.
- Blurry/dark/distant images.

Do not use the external set for threshold selection or model tuning.

## 5.4 Data split

Preferred:

```text
source group A → training
source group B → validation
source group C → public test
```

If source grouping is impossible, use stratified 70/15/15 and state the limitation. Do not claim perfect generalization.

---

# 6. Model plan

## 6.1 Baseline

Train a small CNN if time allows. Its purpose is to establish a baseline.

## 6.2 Main model

Use MobileNetV2 or EfficientNet-B0 transfer learning:

1. Load ImageNet-pretrained backbone.
2. Freeze backbone.
3. Add global average pooling.
4. Add dropout.
5. Add one sigmoid output.
6. Train classification head.
7. Unfreeze final layers.
8. Fine-tune with a low learning rate.

Recommended initial values:

```text
image_size: 224 × 224
batch_size: 32
head_learning_rate: 1e-3
fine_tune_learning_rate: 1e-5
loss: binary cross entropy
optimizer: Adam
seed: 42
```

## 6.3 Augmentation

Apply to training only:

- ±10–15° rotation.
- Mild brightness/contrast change.
- Small zoom/crop.
- Mild translation.
- Mild blur/compression variation.
- Horizontal flip only when physically reasonable.

Do not augment evaluation images.

---

# 7. Evaluation requirements

## 7.1 Overall metrics

Report separately on the standard test set:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Number of false positives.
- Number of false negatives.

## 7.2 Mandatory hard-negative false-positive rate

Calculate on hard negatives alone:

```text
hard_negative_FPR = hard_negative_images_predicted_as_crack
                    / total_hard_negative_images
```

Also calculate:

```text
stain_FPR
joint_FPR
shadow_FPR
```

Required table:

| Set | Total | False positives | FPR |
|---|---:|---:|---:|
| All hard negatives |  |  |  |
| Stains |  |  |  |
| Joints |  |  |  |
| Shadows |  |  |  |

Do not merge this value into overall accuracy. Label it exactly:

> Hard-negative false-positive rate — separate from overall accuracy

## 7.3 Cost-based threshold

Define relative costs:

```text
C_FN = 10  # missed visible crack
C_FP = 1   # unnecessary inspection
```

These are relative planning costs, not financial or engineering claims.

For each candidate threshold:

```text
expected_cost = C_FN × FN + C_FP × FP
normalized_cost = expected_cost / number_of_samples
```

Use only the validation set to select the threshold. Then freeze it before final test evaluation.

The final report must explain:

- A missed crack may delay inspection and create greater risk.
- An unnecessary inspection consumes time and resources.
- Therefore the screening system prioritizes recall, while monitoring hard-negative FPR.
- The chosen threshold is the lowest-cost operating point under the documented relative-cost assumptions, subject to acceptable false-positive behavior.

If cost-minimization produces excessive hard-negative FPR, use an uncertainty band and route uncertain cases to manual review.

## 7.4 Required wrong-case evidence

At least one incorrect prediction must include:

- Original image.
- Ground truth.
- Predicted label.
- Probability.
- Grad-CAM or box evidence.
- Written explanation tied to the displayed evidence.

Example explanation structure:

> This image is a false positive. The model focused on the high-contrast boundary of a shadow, which resembles an elongated crack. The likely reason is that the training data contained fewer shadow examples or that the shadow boundary had similar local texture to a crack. The case should be routed to expert review rather than treated as a structural conclusion.

---

# 8. Colab training implementation

The agent must use the copy-ready script at:

```text
colab/train_crack_model.py
```

The script must support:

```bash
python train_crack_model.py \
  --data-root /content/data \
  --output-root /content/outputs \
  --epochs 12 \
  --batch-size 32
```

The script should also be usable from a Colab notebook by setting `sys.argv` or importing functions.

## 8.1 Expected data layout

```text
/content/data/
├── train/
│   ├── crack/
│   └── no_crack/
├── validation/
│   ├── crack/
│   └── no_crack/
├── test/
│   ├── crack/
│   └── no_crack/
└── hard_negatives/
    ├── stain/
    ├── joint/
    └── shadow/
```

The script may also accept CSV manifests if that is more suitable.

## 8.2 Colab execution sequence

1. Open a new Google Colab notebook.
2. Enable GPU runtime.
3. Install dependencies.
4. Upload or mount the dataset.
5. Copy `train_crack_model.py` into the notebook or upload it.
6. Run the training command.
7. Download:
   - Model.
   - Threshold metadata.
   - Metrics.
   - Heatmaps.
   - Error-case files.
8. Copy artifacts into the project’s `model_artifacts/` and `evaluation_outputs/` folders.

The generated script is provided as a separate artifact with this plan.

---

# 9. Inference API contract

Implement a backend endpoint or adapter with this contract.

## 9.1 Request

```http
POST /predict
Content-Type: multipart/form-data

file=<image file>
```

## 9.2 Response

```json
{
  "label": "potential_crack",
  "probability": 0.87,
  "threshold": 0.60,
  "visual_priority": "medium",
  "quality_status": "acceptable",
  "quality_message": "Image quality is acceptable for visual screening.",
  "review_recommendation": "Submit for expert inspection.",
  "gradcam_url": "/artifacts/gradcam/example.png",
  "limitations": "This is not a structural safety assessment."
}
```

## 9.3 Label rules

```text
probability < threshold - uncertainty_margin
    → no_visible_crack

probability between uncertainty bounds
    → uncertain_manual_review

probability > threshold + uncertainty_margin
    → potential_crack
```

Choose the uncertainty margin from validation results or document a conservative default.

## 9.4 Quality checks

Implement at least:

- Minimum resolution check.
- Blur check using Laplacian variance.
- Brightness check.
- File type and size validation.
- Grayscale/RGBA normalization.

Poor-quality output should lower confidence or route to manual review.

---

# 10. Stitch MCP frontend plan

## 10.1 Tool check

The agent must first check whether Stitch MCP is connected and exposes a frontend-generation/design tool.

If available, use it to create the frontend design and implementation. Preserve the generated code in the repository and continue refining it in the project’s normal frontend structure.

If Stitch MCP is unavailable:

1. Record this in `BUILD_LOG.md`.
2. Use the existing frontend stack if present.
3. Otherwise implement the same design in the repository’s standard React/Vite or Next.js setup.
4. Do not claim the frontend was generated by Stitch.

## 10.2 Stitch MCP design prompt

Give Stitch MCP this exact product brief, adapting only framework-specific details:

> Design and implement a professional responsive web interface called ConcreteGuard for an explainable concrete-surface crack screening assistant. The primary user is a maintenance or inspection team member. The interface must support image upload, drag-and-drop, preview, analysis progress, prediction result, confidence, visual-priority proxy, image-quality warning, Grad-CAM evidence overlay, hard-negative examples, and an expert-review recommendation. The interface must clearly state that it detects visible crack-like image patterns only and does not assess structural safety. Include a clean dashboard layout with a prominent upload card, original-versus-evidence comparison, result status card, confidence meter, evaluation evidence section, and a limitations panel. Include sample/demo images for a crack, stain, joint, shadow, and poor-quality image. Use accessible contrast, keyboard-accessible controls, responsive layout, clear empty/loading/error/success states, and no misleading green ‘safe’ status. Use ‘No visible crack detected’ instead of ‘Safe’. Use ‘Potential crack detected’ instead of ‘Danger confirmed’. The UI should look suitable for a college AI/ML hackathon demo while remaining professional and technically credible.

## 10.3 Required pages/views

### View A — Screening dashboard

- Header and product statement.
- Upload card.
- Drag-and-drop area.
- Image preview.
- Analyze button.
- Capture guidance.
- Safety boundary note.

### View B — Result state

- Original image.
- Grad-CAM overlay.
- Prediction label.
- Probability/confidence.
- Threshold used.
- Visual-priority proxy.
- Image-quality status.
- Expert-review recommendation.
- Button to analyze another image.

### View C — Evidence and evaluation

- Overall metrics cards.
- Hard-negative-only FPR card.
- Stain/joint/shadow FPR table.
- Baseline versus improved model comparison.
- Example failure case.
- Explanation of why the model likely failed.

### View D — About and limitations

- Dataset sources.
- License/attribution.
- Model approach.
- What the system can do.
- What it cannot do.
- Responsible-use statement.

## 10.4 Required frontend states

Implement and visually test:

- Empty state.
- Drag-over state.
- Invalid file state.
- Uploading state.
- Analyzing state.
- Successful crack result.
- Successful no-crack result.
- Uncertain result.
- Poor-quality warning.
- API failure state.
- Offline/demo mode state.
- Mobile responsive layout.

## 10.5 Frontend design constraints

- Do not use a green checkmark with the word `safe`.
- Avoid alarmist red danger messaging.
- Use neutral, evidence-oriented wording.
- Make the result understandable without reading a technical paper.
- Show the model’s limitations near the result, not hidden in a footer only.
- Make Grad-CAM visually comparable to the original image.
- Ensure image URLs and generated artifacts are accessible and do not expose secrets.

## 10.6 Frontend integration tasks

After Stitch MCP generates the UI, the agent must:

1. Inspect generated components.
2. Replace placeholder predictions with API calls or a typed mock adapter.
3. Add a typed response schema.
4. Add loading and error handling.
5. Add demo data for offline presentation.
6. Connect Grad-CAM image rendering.
7. Connect metrics and hard-negative evidence.
8. Validate responsive behavior.
9. Run the frontend build.
10. Fix all build and type errors.

---

# 11. Frontend component contract

Use components equivalent to:

```text
AppShell
Header
UploadDropzone
ImagePreview
AnalysisProgress
PredictionResultCard
ConfidenceMeter
VisualPriorityBadge
ImageQualityAlert
EvidenceComparison
GradCamViewer
HardNegativeSummary
FailureCaseCard
MetricsPanel
LimitationsPanel
DemoImageSelector
FooterDisclaimer
```

Suggested types:

```ts
type PredictionLabel =
  | 'potential_crack'
  | 'no_visible_crack'
  | 'uncertain_manual_review'
  | 'image_quality_insufficient';

type PredictionResponse = {
  label: PredictionLabel;
  probability: number;
  threshold: number;
  visual_priority: 'low' | 'medium' | 'high' | 'manual_review';
  quality_status: 'acceptable' | 'warning' | 'insufficient';
  quality_message: string;
  review_recommendation: string;
  original_url?: string;
  gradcam_url?: string;
  limitations: string;
};
```

---

# 12. Demo experience

The final demo must work in this sequence:

## Demo 1 — Clear crack

1. Select or upload a crack image.
2. Click Analyze.
3. Show `Potential crack detected`.
4. Show probability and threshold.
5. Show Grad-CAM aligned with the crack.
6. Show `Submit for expert inspection`.

## Demo 2 — Stain or shadow

1. Select a hard-negative image.
2. Show `No visible crack detected` or `Uncertain`.
3. Show evidence overlay.
4. Explain hard-negative FPR and why this category is tested separately.

## Demo 3 — Poor-quality image

1. Select a blurry/dark image.
2. Show image-quality warning.
3. Explain why the system should not issue an overconfident result.

## Demo 4 — Model failure

1. Open the saved failure-case card.
2. Show ground truth and model prediction.
3. Show Grad-CAM/box evidence.
4. Explain the likely visual confusion.
5. Explain that the correct action is manual review.

Keep screenshots and local demo fixtures so the presentation does not depend on network access.

---

# 13. Testing plan

## 13.1 Model tests

- Test preprocessing on RGB, grayscale, and RGBA images.
- Test corrupt image handling.
- Test threshold metadata loading.
- Test standard metrics against a small known fixture.
- Test hard-negative FPR formula manually.
- Test Grad-CAM output dimensions.
- Test a known crack example.
- Test a known stain/joint/shadow example.

## 13.2 Backend tests

- `POST /predict` with valid image.
- Invalid file type.
- Oversized file.
- Missing file.
- Model not found.
- Grad-CAM generation failure.
- Poor-quality image.
- Concurrent requests if applicable.

## 13.3 Frontend tests

- Upload works.
- Drag/drop works.
- Loading state appears.
- Error state appears.
- Result card renders all fields.
- Grad-CAM can be viewed.
- Mobile layout does not overflow.
- Keyboard navigation works.
- Demo mode works without backend.

## 13.4 Build verification

Run:

```bash
# frontend
npm install
npm run build

# backend, if present
python -m compileall inference

# training script syntax
python -m py_compile colab/train_crack_model.py
```

Use the actual project commands if different.

---

# 14. Documentation requirements

## README.md

Include:

- What the project does.
- What it does not do.
- Architecture diagram.
- Local setup.
- Colab setup.
- Frontend setup.
- Inference API.
- Demo instructions.
- Evaluation summary.
- Dataset attribution.

## DATASET_CARD.md

Include:

- Dataset URLs.
- Licenses.
- Class counts.
- Hard-negative collection method.
- Split strategy.
- Leakage limitations.
- External test methodology.

## MODEL_CARD.md

Include:

- Intended use.
- Out-of-scope use.
- Architecture.
- Input constraints.
- Threshold.
- Metrics.
- Hard-negative FPR.
- Known failure modes.
- Human-review recommendation.

## PHASE1_REPORT.md

Include:

1. Overall metrics.
2. Separate hard-negative FPR.
3. Per-category stain/joint/shadow FPR.
4. Cost assumptions.
5. Threshold justification.
6. Correct Grad-CAM example.
7. Wrong Grad-CAM/box example and written cause analysis.
8. Limitations.

---

# 15. Final acceptance criteria

The build is complete only if all are true:

## Model and data

- [ ] Dataset source and license documented.
- [ ] Crack/no-crack pipeline runs.
- [ ] Hard negatives include stains, joints, and shadows.
- [ ] External test set is separate.
- [ ] Model checkpoint is saved.
- [ ] Inference preprocessing is reusable.

## Evaluation

- [ ] Accuracy reported.
- [ ] Precision reported.
- [ ] Recall reported.
- [ ] F1 reported.
- [ ] Confusion matrix saved.
- [ ] Hard-negative-only FPR reported separately.
- [ ] Stain FPR reported.
- [ ] Joint FPR reported.
- [ ] Shadow FPR reported.
- [ ] Cost model documented.
- [ ] Threshold selected using validation data.
- [ ] Threshold rationale written.
- [ ] At least one wrong case shown.
- [ ] Wrong case includes Grad-CAM or boxes.
- [ ] Wrong case has an image-specific explanation.

## Frontend

- [ ] Stitch MCP used if available, or fallback documented.
- [ ] Upload state works.
- [ ] Result state works.
- [ ] Grad-CAM displays.
- [ ] Confidence and threshold display.
- [ ] Visual-priority wording is cautious.
- [ ] Hard-negative evidence appears.
- [ ] Limitations are visible.
- [ ] Demo mode works offline.
- [ ] Frontend builds successfully.

## Handoff

- [ ] README complete.
- [ ] Colab script complete.
- [ ] Model artifacts included or download instructions provided.
- [ ] API contract documented.
- [ ] Environment variables documented without secrets.
- [ ] Build log updated.

---

# 16. Complete agent prompt

Copy the following prompt to the implementation agent:

> Build the complete JIGNASA concrete surface crack inspection project from the attached master plan. First inspect the existing repository and preserve working code. Implement the data pipeline, model training/evaluation artifacts, inference contract, and frontend. Use Google Colab-compatible Python training code for MobileNetV2 transfer learning with training-only augmentation. The model must classify visible crack versus no visible crack. Create and preserve a separate hard-negative set for stains, construction joints, and shadows. Calculate standard accuracy, precision, recall, F1, confusion matrix, and a hard-negative-only false-positive rate computed only on the hard-negative images. Report stain, joint, and shadow FPR separately. Select and document a decision threshold using validation-only cost analysis where a missed crack has a higher relative cost than an unnecessary inspection. Generate a threshold-versus-cost plot. Implement Grad-CAM and save at least one correct example and one incorrect example. The incorrect example must include the original image, Grad-CAM or box evidence, prediction, ground truth, and a written image-specific explanation of why the model likely failed. Export the trained model, preprocessing metadata, threshold metadata, metrics, manifests, and evaluation artifacts. For the frontend, check whether Stitch MCP is available. If available, use it to generate a polished responsive upload-and-review interface with upload, preview, analysis progress, prediction result, confidence, visual-priority proxy, image-quality warning, original-versus-Grad-CAM comparison, hard-negative evidence, failure-case evidence, metrics, and visible limitations. If Stitch MCP is unavailable, document that and implement the same interface using the existing frontend stack or a standard React/Vite fallback. Connect the frontend to a live inference endpoint if available; otherwise provide a clearly labeled offline/demo adapter. Do not claim structural safety or structural severity. Run all build, syntax, and smoke tests. Update README.md, DATASET_CARD.md, MODEL_CARD.md, PHASE1_REPORT.md, BUILD_LOG.md, and provide an exact summary of files, commands, metrics, known limitations, and demo instructions. Do not invent metrics or claim tools were used when they were unavailable.
