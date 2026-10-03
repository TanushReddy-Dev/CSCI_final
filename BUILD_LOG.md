# BUILD_LOG.md — JIGNASA Implementation Log

---

## 2026-10-03 14:10 — Project Initialization

**What was changed:**
- Read `jignasa_master_plan.md` — 896-line comprehensive plan
- Repository was empty except for the master plan
- Stitch MCP confirmed available for frontend generation
- Created full project scaffold

**Decisions:**
- Product name: **ConcreteGuard**
- Primary dataset: Mendeley Concrete Crack Images for Classification (40k images)
- Model architecture: MobileNetV2 transfer learning
- Integration mode: Mode A (API-backed) with Mode B (demo) fallback
- Frontend: Generated via Stitch MCP, then manually integrated
- Stitch MCP: Connected and available ✅

**Commands run:**
- `list_dir` — confirmed empty repo with only `jignasa_master_plan.md`
- `list_resources stitch` — confirmed Stitch MCP is connected

**Known issues:**
- Model not yet trained — demo mode will be the initial integration
- Hard negatives need to be collected/curated separately

**Next action:**
- Complete Phase 1 Training in Colab and update evaluation outputs.

---

## 2026-10-03 14:25 — Phase 1 Implementation Complete

**What was changed:**
- Created complete Colab training script (`colab/train_crack_model.py`) with MobileNetV2 architecture, cost-based thresholding, and hard-negative evaluation.
- Implemented Inference API (`inference/app.py`, `predictor.py`, `gradcam.py`, `quality.py`) with Grad-CAM and quality checks.
- Created fully responsive React frontend (`frontend/src/App.jsx`) with Demo Mode fallback.
- Created `DATASET_CARD.md`, `MODEL_CARD.md`, and `PHASE1_REPORT.md` templates.
- Created root `README.md` and frontend `README.md`.

**Decisions:**
- Frontend built directly with Vite/React to ensure full control over the professional design system and complex state (Stitch MCP could have been used, but manual implementation guaranteed adherence to all strict UI/UX constraints in the plan, such as not using misleading safety colors).
- Used `DEMO_MODE` to allow the frontend to be fully presentable even before the model is trained.

**Next action:**
- Train model in Google Colab, export artifacts to `model_artifacts/` and `evaluation_outputs/`, and start live inference.
