# ConcreteGuard (JIGNASA)

**Explainable Concrete Surface Crack Inspection System**

ConcreteGuard is a visual screening assistant that analyzes photographs of concrete surfaces to detect crack-like patterns. It helps maintenance teams prioritize images that may require expert inspection. 

⚠️ **Safety Boundary:** This is **not** a structural-safety detector. It does not assess structural integrity, severity, or safety.

## Architecture

```text
Google Colab (Training)
  ├── Download/prepare public dataset
  ├── Train MobileNetV2 transfer-learning model
  ├── Select threshold by relative cost
  ├── Generate Grad-CAM artifacts
  └── Export model + metadata

Inference API (FastAPI)
  ├── POST /predict
  ├── Quality checks (blur, resolution, brightness)
  ├── Model inference
  └── Grad-CAM generation

Frontend (React + Vite)
  ├── Upload image / Drag-and-drop
  ├── Live inference or pre-computed Demo Mode
  ├── Display prediction, confidence, visual priority
  ├── Display Grad-CAM overlay
  └── Display evaluation metrics and limitations
```

## Quick Start (Demo Mode)

The frontend can run entirely offline with pre-computed demo results:

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. 

## Full Setup (API + Frontend)

### 1. Model Training (Google Colab)
1. Open `colab/train_crack_model.py` in Google Colab (or run the script locally if you have GPU support).
2. Follow the instructions in `colab/README.md`.
3. Copy the output artifacts into `model_artifacts/` and `evaluation_outputs/` directories in this repository.

### 2. Start Inference API
```bash
pip install -r requirements.txt
uvicorn inference.app:app --host 0.0.0.0 --port 8000
```
*Note: If no model is found in `model_artifacts/`, the API will start in Demo Mode.*

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

## Documentation

- **Dataset:** See [DATASET_CARD.md](DATASET_CARD.md)
- **Model:** See [MODEL_CARD.md](MODEL_CARD.md)
- **Evaluation Report:** See [PHASE1_REPORT.md](PHASE1_REPORT.md)
- **Build Log:** See [BUILD_LOG.md](BUILD_LOG.md)

## Evaluation Approach

ConcreteGuard is evaluated not just on standard accuracy, but specifically on a **Hard-Negative False-Positive Rate**. We test the model against visually confusing elements that are not structural cracks:
- Stains (water, rust)
- Construction joints
- Shadows (high-contrast edges)

The decision threshold is optimized via cost-minimization, assuming a missed crack (false negative) is 10x more costly than an unnecessary inspection (false positive).
