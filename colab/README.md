# ConcreteGuard Colab Training

## Quick Start

1. Open Google Colab and enable **GPU runtime** (Runtime → Change runtime type → T4 GPU)
2. Upload `train_crack_model.py` to the Colab environment
3. Install dependencies:

```bash
!pip install tensorflow pandas scikit-learn matplotlib seaborn
```

4. Mount Google Drive and ensure your dataset is located at `/content/drive/MyDrive/SDNET2018.zip`.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. Run training:

### Recommended Fast Run (High Performance on T4 GPU)
```bash
!python train_crack_model.py \
    --zip-path /content/drive/MyDrive/SDNET2018.zip \
    --data-root /content/data \
    --output-root /content/outputs \
    --epochs 8 \
    --head-epochs 3 \
    --batch-size 64
```

### Ultra-Fast Smoke Test (~3–4 Minutes)
To quickly verify everything works end-to-end without waiting for all 56,000 images:
```bash
!python train_crack_model.py \
    --zip-path /content/drive/MyDrive/SDNET2018.zip \
    --data-root /content/data \
    --output-root /content/outputs \
    --quick-run
```

The script will automatically check for GPU availability, extract the zip file, parse the SDNET2018 structure, and train using multi-threaded data loading.

6. Download outputs from `/content/outputs/` — includes model, metrics, Grad-CAM, and evaluation artifacts.

## Using from a Notebook Cell

```python
import sys
sys.argv = ['train_crack_model.py',
            '--data-root', '/content/data',
            '--output-root', '/content/outputs',
            '--epochs', '8',
            '--batch-size', '64']
exec(open('train_crack_model.py').read())
```

## Expected Outputs

```
outputs/
├── model_artifacts/
│   ├── best_model.keras
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
│   ├── correct_case_original.png
│   ├── correct_case_gradcam.png
│   └── sample_predictions.csv
└── data_manifests/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── hard_negatives.csv
```
