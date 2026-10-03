#!/usr/bin/env python3
"""
ConcreteGuard — Concrete Surface Crack Classification Training Script
=====================================================================

MobileNetV2 transfer-learning pipeline for binary crack / no-crack classification.
Designed to run in Google Colab with GPU runtime.

Usage (CLI):
    python train_crack_model.py \
        --data-root /content/data \
        --output-root /content/outputs \
        --epochs 12 \
        --batch-size 32

Usage (Colab notebook):
    import sys
    sys.argv = ['train_crack_model.py',
                '--data-root', '/content/data',
                '--output-root', '/content/outputs']
    exec(open('train_crack_model.py').read())

    # Or import individual functions:
    from train_crack_model import build_model, train, evaluate
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224)
SEED = 42
CLASS_NAMES = ["no_crack", "crack"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConcreteGuard crack classifier")
    parser.add_argument("--data-root", type=str, default="/content/data",
                        help="Root directory with train/validation/test/hard_negatives subdirs")
    parser.add_argument("--zip-path", type=str, default="/content/drive/MyDrive/SDNET2018.zip",
                        help="Path to SDNET2018.zip file to extract and format")
    parser.add_argument("--output-root", type=str, default="/content/outputs",
                        help="Directory for model artifacts and evaluation outputs")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Total training epochs (head + fine-tune)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training and evaluation")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="Learning rate for classification head training")
    parser.add_argument("--finetune-lr", type=float, default=1e-5,
                        help="Learning rate for fine-tuning phase")
    parser.add_argument("--head-epochs", type=int, default=3,
                        help="Epochs for head-only training before fine-tuning")
    parser.add_argument("--cfn", type=float, default=10.0,
                        help="Relative cost of false negative (missed crack)")
    parser.add_argument("--cfp", type=float, default=1.0,
                        help="Relative cost of false positive (unnecessary inspection)")
    parser.add_argument("--uncertainty-margin", type=float, default=0.10,
                        help="Margin around threshold for uncertain predictions")
    parser.add_argument("--train-baseline", action="store_true",
                        help="Also train a simple CNN baseline for comparison")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Limit number of batches per training epoch")
    parser.add_argument("--validation-steps", type=int, default=None,
                        help="Limit number of validation batches")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap maximum images per class during extraction (e.g. 5000)")
    parser.add_argument("--quick-run", action="store_true",
                        help="Fast smoke test: caps steps per epoch for a ~3 minute run")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def prepare_sdnet2018(zip_path: str, data_root: str, max_samples: int = None):
    """Extract and restructure SDNET2018 dataset."""
    import zipfile
    import shutil
    import random
    
    if os.path.exists(os.path.join(data_root, "train")):
        print("  Data already structured in data-root. Skipping extraction.")
        return

    extract_dir = os.path.join(data_root, "raw_extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"  Extracting zip from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except Exception as e:
        print(f"  ❌ Error extracting zip: {e}")
        return

    print("  Restructuring into train/validation/test...")
    all_crack = []
    all_nocrack = []
    
    for root, dirs, files in os.walk(extract_dir):
        parent_dir = os.path.basename(root)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if parent_dir.startswith('C'):
                    all_crack.append(os.path.join(root, f))
                elif parent_dir.startswith('U'):
                    all_nocrack.append(os.path.join(root, f))
                
    random.seed(SEED)
    random.shuffle(all_crack)
    random.shuffle(all_nocrack)
    
    if max_samples and max_samples > 0:
        print(f"  Capping dataset at {max_samples} samples per class...")
        all_crack = all_crack[:max_samples]
        all_nocrack = all_nocrack[:max_samples]
    
    # 70/15/15 split
    def split_data(files):
        n = len(files)
        tr = int(n * 0.7)
        va = int(n * 0.15)
        return files[:tr], files[tr:tr+va], files[tr+va:]
        
    crack_tr, crack_va, crack_te = split_data(all_crack)
    nocrack_tr, nocrack_va, nocrack_te = split_data(all_nocrack)
    
    mapping = {
        "train": {"crack": crack_tr, "no_crack": nocrack_tr},
        "validation": {"crack": crack_va, "no_crack": nocrack_va},
        "test": {"crack": crack_te, "no_crack": nocrack_te}
    }
    
    for split in mapping:
        for cls in mapping[split]:
            os.makedirs(os.path.join(data_root, split, cls), exist_ok=True)
            for f in mapping[split][cls]:
                # Rename file slightly to avoid collision if needed, but basenames should be unique enough in SDNET
                shutil.copy2(f, os.path.join(data_root, split, cls, os.path.basename(f)))
                
    print(f"  Moved {len(all_crack)} crack and {len(all_nocrack)} no_crack images.")


def create_data_generators(data_root: str, batch_size: int):
    """Create train (with augmentation), validation, and test generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    generators = {}

    train_dir = os.path.join(data_root, "train")
    if os.path.isdir(train_dir):
        generators["train"] = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode="binary",
            classes=CLASS_NAMES,
            shuffle=True,
            seed=SEED,
        )

    for split in ["validation", "test"]:
        split_dir = os.path.join(data_root, split)
        if os.path.isdir(split_dir):
            generators[split] = eval_datagen.flow_from_directory(
                split_dir,
                target_size=IMG_SIZE,
                batch_size=batch_size,
                class_mode="binary",
                classes=CLASS_NAMES,
                shuffle=False,
                seed=SEED,
            )

    return generators


def load_hard_negatives(data_root: str, batch_size: int):
    """Load hard-negative images (stains, joints, shadows) — all are ground truth no_crack."""
    hn_dir = os.path.join(data_root, "hard_negatives")
    if not os.path.isdir(hn_dir):
        print(f"[WARN] Hard-negatives directory not found: {hn_dir}")
        return None, {}

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    categories = {}
    all_images = []
    all_paths = []

    for category in sorted(os.listdir(hn_dir)):
        cat_dir = os.path.join(hn_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        files = [f for f in os.listdir(cat_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        for f in files:
            fpath = os.path.join(cat_dir, f)
            all_paths.append(fpath)
            categories[fpath] = category

    if not all_paths:
        print("[WARN] No hard-negative images found")
        return None, {}

    # Load images
    images = []
    valid_paths = []
    for p in all_paths:
        try:
            img = keras.utils.load_img(p, target_size=IMG_SIZE)
            arr = keras.utils.img_to_array(img) / 255.0
            images.append(arr)
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    images_array = np.array(images)
    path_categories = {p: categories[p] for p in valid_paths}

    return images_array, path_categories


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_mobilenetv2(input_shape=(224, 224, 3), dropout_rate=0.3):
    """Build MobileNetV2 transfer-learning model."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False  # Freeze backbone initially

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ], name="ConcreteGuard_MobileNetV2")

    return model, base


def build_baseline_cnn(input_shape=(224, 224, 3)):
    """Build a simple CNN baseline for comparison."""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="ConcreteGuard_Baseline_CNN")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, base_model, generators, args, output_root):
    """Two-phase training: head-only → fine-tune."""
    model_dir = os.path.join(output_root, "model_artifacts")
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, "best_model.keras")

    common_callbacks = [
        callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-7, verbose=1
        ),
    ]

    # Phase 1: Train classification head only
    print("\n" + "=" * 60)
    print("PHASE 1: Training classification head (backbone frozen)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.head_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    head_epochs = min(args.head_epochs, args.epochs)
    history_head = model.fit(
        generators["train"],
        validation_data=generators["validation"],
        epochs=head_epochs,
        callbacks=common_callbacks,
        steps_per_epoch=getattr(args, "steps_per_epoch", None),
        validation_steps=getattr(args, "validation_steps", None),
    )

    # Phase 2: Fine-tune last layers of backbone
    remaining_epochs = args.epochs - head_epochs
    if remaining_epochs > 0 and base_model is not None:
        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning (unfreezing last 30 layers)")
        print("=" * 60)

        # Unfreeze last 30 layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.finetune_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        history_ft = model.fit(
            generators["train"],
            validation_data=generators["validation"],
            initial_epoch=head_epochs,
            epochs=args.epochs,
            callbacks=common_callbacks,
            steps_per_epoch=getattr(args, "steps_per_epoch", None),
            validation_steps=getattr(args, "validation_steps", None),
        )
    else:
        history_ft = None

    # Reload best checkpoint
    if os.path.exists(checkpoint_path):
        model = keras.models.load_model(checkpoint_path)
        print(f"\n✅ Loaded best model from {checkpoint_path}")

    return model, history_head, history_ft


def train_baseline(generators, args, output_root):
    """Train and evaluate the simple CNN baseline."""
    print("\n" + "=" * 60)
    print("BASELINE: Training simple CNN for comparison")
    print("=" * 60)

    baseline = build_baseline_cnn()
    baseline.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.head_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    baseline_dir = os.path.join(output_root, "baseline_artifacts")
    os.makedirs(baseline_dir, exist_ok=True)

    baseline.fit(
        generators["train"],
        validation_data=generators["validation"],
        epochs=min(5, args.epochs),
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ],
        steps_per_epoch=getattr(args, "steps_per_epoch", None),
        validation_steps=getattr(args, "validation_steps", None),
    )

    # Evaluate baseline
    test_loss, test_acc = baseline.evaluate(generators["test"], verbose=0)
    baseline_metrics = {"test_accuracy": float(test_acc), "test_loss": float(test_loss)}

    with open(os.path.join(baseline_dir, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    print(f"Baseline test accuracy: {test_acc:.4f}")
    return baseline_metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, generators, args, output_root):
    """Evaluate on test set, compute metrics, confusion matrix, and cost-based threshold."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix as sk_confusion_matrix, classification_report
    )

    eval_dir = os.path.join(output_root, "evaluation_outputs")
    os.makedirs(eval_dir, exist_ok=True)

    # Predictions on test set
    test_gen = generators["test"]
    test_gen.reset()
    y_true = test_gen.classes
    print("  Computing test set predictions...")
    y_prob = model.predict(test_gen, verbose=1).flatten()

    # --- Cost-based threshold selection on VALIDATION set ---
    val_gen = generators["validation"]
    val_gen.reset()
    val_true = val_gen.classes
    print("  Computing validation set predictions (for cost threshold)...")
    val_prob = model.predict(val_gen, verbose=1).flatten()

    thresholds = np.arange(0.05, 0.96, 0.01)
    costs = []
    for t in thresholds:
        val_pred = (val_prob >= t).astype(int)
        fn = np.sum((val_pred == 0) & (val_true == 1))
        fp = np.sum((val_pred == 1) & (val_true == 0))
        cost = args.cfn * fn + args.cfp * fp
        normalized_cost = cost / len(val_true)
        costs.append({
            "threshold": float(t),
            "FN": int(fn), "FP": int(fp),
            "cost": float(cost),
            "normalized_cost": float(normalized_cost),
        })

    costs_df = pd.DataFrame(costs)
    best_idx = costs_df["normalized_cost"].idxmin()
    best_threshold = costs_df.loc[best_idx, "threshold"]

    print(f"\n✅ Optimal threshold (validation cost-minimization): {best_threshold:.2f}")

    # Save threshold
    threshold_info = {
        "threshold": float(best_threshold),
        "uncertainty_margin": args.uncertainty_margin,
        "C_FN": args.cfn,
        "C_FP": args.cfp,
        "selected_on": "validation_set",
        "selection_method": "cost_minimization",
    }
    model_dir = os.path.join(output_root, "model_artifacts")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)

    # --- Evaluate on test set with chosen threshold ---
    y_pred = (y_prob >= best_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = sk_confusion_matrix(y_true, y_pred)

    tn, fp_count, fn_count, tp = cm.ravel()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "threshold": float(best_threshold),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp_count),
            "FN": int(fn_count), "TP": int(tp),
        },
        "total_samples": int(len(y_true)),
        "false_positives": int(fp_count),
        "false_negatives": int(fn_count),
    }

    # Save metrics
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

    # Save cost curve
    costs_df.to_csv(os.path.join(eval_dir, "threshold_cost_data.csv"), index=False)

    # Plot confusion matrix
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Test Set)")
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)

        # Plot threshold-cost curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(costs_df["threshold"], costs_df["normalized_cost"], "b-", linewidth=2)
        ax.axvline(best_threshold, color="red", linestyle="--", label=f"Selected: {best_threshold:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Normalized Cost")
        ax.set_title(f"Threshold vs. Cost (C_FN={args.cfn}, C_FP={args.cfp})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "threshold_cost_curve.png"), dpi=150)
        plt.close(fig)

        print("✅ Saved confusion matrix and threshold-cost curve plots")
    except ImportError:
        print("[WARN] matplotlib/seaborn not available — skipping plots")

    # Save sample predictions
    sample_df = pd.DataFrame({
        "index": range(len(y_true)),
        "ground_truth": [CLASS_NAMES[int(y)] for y in y_true],
        "predicted": [CLASS_NAMES[int(y)] for y in y_pred],
        "probability": y_prob,
        "correct": y_true == y_pred,
    })
    sample_df.to_csv(os.path.join(eval_dir, "sample_predictions.csv"), index=False)

    print(f"\n{'='*60}")
    print("TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Threshold:  {best_threshold:.2f}")
    print(f"TP={tp}  FP={fp_count}  FN={fn_count}  TN={tn}")

    return metrics, y_prob, y_true, y_pred, best_threshold


def evaluate_hard_negatives(model, data_root, output_root, threshold):
    """Evaluate model on hard-negative images and compute per-category FPR."""
    eval_dir = os.path.join(output_root, "evaluation_outputs")
    os.makedirs(eval_dir, exist_ok=True)

    images, path_categories = load_hard_negatives(data_root, batch_size=32)
    if images is None:
        print("[WARN] Skipping hard-negative evaluation — no images found")
        return None

    probs = model.predict(images, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)

    paths = list(path_categories.keys())
    cats = [path_categories[p] for p in paths]

    results = pd.DataFrame({
        "image_path": paths,
        "category": cats,
        "ground_truth": "no_crack",
        "probability": probs,
        "predicted": [CLASS_NAMES[int(p)] for p in preds],
        "false_positive": preds.astype(bool),
    })
    results.to_csv(os.path.join(eval_dir, "hard_negative_results.csv"), index=False)

    # Per-category FPR
    summary_rows = []
    for cat in sorted(results["category"].unique()):
        cat_data = results[results["category"] == cat]
        cat_fp = cat_data["false_positive"].sum()
        cat_total = len(cat_data)
        cat_fpr = cat_fp / cat_total if cat_total > 0 else 0
        summary_rows.append({
            "category": cat,
            "total": int(cat_total),
            "false_positives": int(cat_fp),
            "FPR": float(cat_fpr),
        })

    # Overall hard-negative FPR
    total_fp = results["false_positive"].sum()
    total_count = len(results)
    overall_fpr = total_fp / total_count if total_count > 0 else 0

    summary_rows.insert(0, {
        "category": "all_hard_negatives",
        "total": int(total_count),
        "false_positives": int(total_fp),
        "FPR": float(overall_fpr),
    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(eval_dir, "hard_negative_summary.csv"), index=False)

    hn_summary = {
        "overall_FPR": float(overall_fpr),
        "total_images": int(total_count),
        "total_false_positives": int(total_fp),
        "per_category": {row["category"]: row for _, row in summary.iterrows()},
    }
    with open(os.path.join(eval_dir, "hard_negative_summary.json"), "w") as f:
        json.dump(hn_summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("HARD-NEGATIVE FALSE-POSITIVE RATE — separate from overall accuracy")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    return hn_summary


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def generate_gradcam(model, img_array, layer_name=None):
    """Generate Grad-CAM heatmap for a single image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, 224, 224, 3), scaled to [0, 1]
        layer_name: Name of the convolutional layer to use. If None, auto-detect.
    
    Returns:
        heatmap: numpy array (224, 224) with values in [0, 1]
    """
    try:
        img_tensor = tf.cast(img_array, tf.float32)

        # Handle Sequential model containing MobileNetV2 base
        if hasattr(model, "layers") and len(model.layers) > 1 and hasattr(model.layers[0], "layers"):
            base = model.layers[0]
            target_layer = None
            if layer_name:
                try:
                    target_layer = base.get_layer(layer_name)
                except Exception:
                    pass
            if target_layer is None:
                for l in reversed(base.layers):
                    if "out_relu" in l.name.lower() or "conv" in l.name.lower() or isinstance(l, (layers.Conv2D, tf.keras.layers.Conv2D)):
                        target_layer = l
                        break

            if target_layer is None:
                target_layer = base.layers[-1]

            feat_model = keras.Model(inputs=base.inputs, outputs=target_layer.output)

            with tf.GradientTape() as tape:
                conv_outputs = feat_model(img_tensor)
                tape.watch(conv_outputs)
                x = conv_outputs
                for lyr in model.layers[1:]:
                    x = lyr(x)
                loss = x[:, 0]

            grads = tape.gradient(loss, conv_outputs)
        else:
            # Flat model or simple CNN
            target_layer = None
            if layer_name:
                try:
                    target_layer = model.get_layer(layer_name)
                except Exception:
                    pass
            if target_layer is None:
                for l in reversed(model.layers):
                    if isinstance(l, (layers.Conv2D, tf.keras.layers.Conv2D)) or "conv" in l.name.lower():
                        target_layer = l
                        break

            if target_layer is None:
                target_layer = model.layers[-2]

            inputs = getattr(model, "inputs", None)
            if inputs is None and hasattr(model, "input"):
                inputs = model.input

            grad_model = keras.Model(inputs=inputs, outputs=[target_layer.output, model.outputs[0]])
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("[WARN] Grad-CAM gradient is None")
            return np.zeros(IMG_SIZE, dtype=np.float32)

        # Global average pooling of gradients
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1)
        cam = tf.nn.relu(cam).numpy()[0]

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to image size
        cam_resized = tf.image.resize(cam[..., np.newaxis], IMG_SIZE).numpy()[:, :, 0]
        return cam_resized.astype(np.float32)

    except Exception as e:
        print(f"[WARN] Grad-CAM generation error: {e}")
        return np.zeros(IMG_SIZE, dtype=np.float32)


def save_gradcam_overlay(original_img, heatmap, save_path, alpha=0.4):
    """Save Grad-CAM heatmap overlaid on the original image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(original_img)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_error_case(model, generators, threshold, output_root):
    """Find and document at least one incorrect prediction with Grad-CAM evidence."""
    try:
        eval_dir = os.path.join(output_root, "evaluation_outputs")

        test_gen = generators["test"]
        test_gen.reset()

        y_true = test_gen.classes
        y_prob = model.predict(test_gen, verbose=0).flatten()
        y_pred = (y_prob >= threshold).astype(int)

        # Find wrong predictions
        wrong_indices = np.where(y_true != y_pred)[0]
        if len(wrong_indices) == 0:
            print("[INFO] No incorrect predictions found on test set")
            return None

        # Pick the most confident wrong prediction
        wrong_probs = y_prob[wrong_indices]
        most_confident_wrong = wrong_indices[np.argmax(np.abs(wrong_probs - 0.5))]

        # Load the specific image
        idx = most_confident_wrong
        fpath = test_gen.filepaths[idx]
        true_label = CLASS_NAMES[int(y_true[idx])]
        pred_label = CLASS_NAMES[int(y_pred[idx])]
        prob = float(y_prob[idx])

        img = keras.utils.load_img(fpath, target_size=IMG_SIZE)
        img_array = keras.utils.img_to_array(img) / 255.0

        # Generate Grad-CAM
        heatmap = generate_gradcam(model, img_array[np.newaxis])

        # Save original
        img.save(os.path.join(eval_dir, "error_case_original.png"))

        # Save Grad-CAM overlay
        save_gradcam_overlay(img_array, heatmap,
                             os.path.join(eval_dir, "error_case_gradcam.png"))

        # Determine error type and write explanation
        is_false_positive = (true_label == "no_crack" and pred_label == "crack")

        if is_false_positive:
            explanation = (
                f"This image is a FALSE POSITIVE. The model predicted '{pred_label}' "
                f"(probability {prob:.3f}) but the ground truth is '{true_label}'. "
                f"The Grad-CAM heatmap shows the model focused on regions that visually "
                f"resemble crack-like patterns — possibly high-contrast edges, texture "
                f"boundaries, or shadow lines. The likely cause is that these local texture "
                f"features share visual similarity with cracks in the training data. "
                f"The correct action is to route this case for manual expert review rather "
                f"than treating the model's output as a structural conclusion."
            )
        else:
            explanation = (
                f"This image is a FALSE NEGATIVE. The model predicted '{pred_label}' "
                f"(probability {prob:.3f}) but the ground truth is '{true_label}'. "
                f"The Grad-CAM heatmap shows the model did not strongly activate on the "
                f"crack region. The likely cause is that the crack is thin, low-contrast, "
                f"or partially occluded, making it harder to distinguish from normal concrete "
                f"texture. This is the more dangerous error type — a missed crack that should "
                f"have been flagged for inspection. This underscores the importance of using "
                f"the system as a screening aid with human oversight, not a replacement for "
                f"expert inspection."
            )

        error_case = {
            "image_path": fpath,
            "ground_truth": true_label,
            "predicted": pred_label,
            "probability": prob,
            "threshold": float(threshold),
            "error_type": "false_positive" if is_false_positive else "false_negative",
            "explanation": explanation,
            "gradcam_path": "error_case_gradcam.png",
            "original_path": "error_case_original.png",
        }

        with open(os.path.join(eval_dir, "error_case.json"), "w") as f:
            json.dump(error_case, f, indent=2)

        print(f"\n{'='*60}")
        print("ERROR CASE DOCUMENTED")
        print(f"{'='*60}")
        print(f"Type:        {error_case['error_type']}")
        print(f"Ground truth: {true_label}")
        print(f"Predicted:    {pred_label} ({prob:.3f})")
        print(f"Explanation:  {explanation[:200]}...")

        return error_case
    except Exception as e:
        print(f"[WARN] Error in generate_error_case: {e}")
        return None


def generate_correct_gradcam(model, generators, threshold, output_root):
    """Save a Grad-CAM example for a correct crack prediction."""
    try:
        eval_dir = os.path.join(output_root, "evaluation_outputs")

        test_gen = generators["test"]
        test_gen.reset()

        y_true = test_gen.classes
        y_prob = model.predict(test_gen, verbose=0).flatten()
        y_pred = (y_prob >= threshold).astype(int)

        # Find correct crack predictions (true positives)
        tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
        if len(tp_indices) == 0:
            print("[WARN] No true positive predictions for Grad-CAM sample")
            return

        # Pick the most confident true positive
        tp_probs = y_prob[tp_indices]
        best_tp = tp_indices[np.argmax(tp_probs)]

        fpath = test_gen.filepaths[best_tp]
        img = keras.utils.load_img(fpath, target_size=IMG_SIZE)
        img_array = keras.utils.img_to_array(img) / 255.0

        heatmap = generate_gradcam(model, img_array[np.newaxis])
        save_gradcam_overlay(img_array, heatmap,
                             os.path.join(eval_dir, "correct_case_gradcam.png"))
        img.save(os.path.join(eval_dir, "correct_case_original.png"))

        print("✅ Saved correct-case Grad-CAM example")
    except Exception as e:
        print(f"[WARN] Error in generate_correct_gradcam: {e}")


# ---------------------------------------------------------------------------
# Metadata export
# ---------------------------------------------------------------------------

def save_metadata(output_root, args):
    """Save model metadata, class names, and preprocessing info."""
    model_dir = os.path.join(output_root, "model_artifacts")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    preprocessing = {
        "image_size": list(IMG_SIZE),
        "rescale": 1.0 / 255,
        "color_mode": "rgb",
        "interpolation": "bilinear",
    }
    with open(os.path.join(model_dir, "preprocessing.json"), "w") as f:
        json.dump(preprocessing, f, indent=2)

    metadata = {
        "model_name": "ConcreteGuard_MobileNetV2",
        "architecture": "MobileNetV2 + GlobalAvgPool + Dropout + Dense(1, sigmoid)",
        "input_shape": [224, 224, 3],
        "output": "binary sigmoid (crack probability)",
        "class_names": CLASS_NAMES,
        "training_date": datetime.now().isoformat(),
        "training_args": vars(args),
        "framework": "TensorFlow/Keras",
        "intended_use": "Visual screening aid for concrete surface crack detection",
        "not_intended_for": "Structural safety assessment or engineering decisions",
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Saved model metadata, class names, and preprocessing config")


# ---------------------------------------------------------------------------
# Data manifest generation
# ---------------------------------------------------------------------------

def generate_manifests(data_root, output_root):
    """Generate CSV manifests for all data splits."""
    manifest_dir = os.path.join(output_root, "data_manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        rows = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    rows.append({
                        "image_path": os.path.join(split, class_name, fname),
                        "class": class_name,
                        "split": split,
                    })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(manifest_dir, f"{split}.csv"), index=False)
        print(f"  {split}: {len(df)} images")

    # Hard negatives manifest
    hn_dir = os.path.join(data_root, "hard_negatives")
    if os.path.isdir(hn_dir):
        rows = []
        for category in sorted(os.listdir(hn_dir)):
            cat_dir = os.path.join(hn_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in os.listdir(cat_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    rows.append({
                        "image_path": os.path.join("hard_negatives", category, fname),
                        "category": category,
                        "ground_truth": "no_crack",
                        "source": "curated",
                        "lighting": "unknown",
                        "capture_notes": "",
                    })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(manifest_dir, "hard_negatives.csv"), index=False)
        print(f"  hard_negatives: {len(df)} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    if args.quick_run:
        print("⚡ Quick-run mode enabled: capping epochs and steps for fast test run.")
        args.epochs = min(args.epochs, 4)
        args.head_epochs = min(args.head_epochs, 2)
        if args.steps_per_epoch is None:
            args.steps_per_epoch = 100
        if args.validation_steps is None:
            args.validation_steps = 30

    print("=" * 60)
    print("ConcreteGuard — Concrete Crack Classification Training")
    print("=" * 60)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🚀 GPU detected: {len(gpus)} device(s) - {[g.name for g in gpus]}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    else:
        print("⚠️  WARNING: No GPU detected! Running on CPU will take >10 min per epoch.")
        print("    If in Google Colab, go to: Runtime -> Change runtime type -> select T4 GPU.")
    print(f"Data root:        {args.data_root}")
    print(f"Output root:      {args.output_root}")
    print(f"Epochs:           {args.epochs} (head: {args.head_epochs}, fine-tune: {args.epochs - args.head_epochs})")
    print(f"Batch size:       {args.batch_size}")
    if args.steps_per_epoch:
        print(f"Steps/epoch:      {args.steps_per_epoch}")
    if args.validation_steps:
        print(f"Validation steps: {args.validation_steps}")
    print(f"C_FN:             {args.cfn}")
    print(f"C_FP:             {args.cfp}")
    print()

    # Create output directories
    os.makedirs(args.output_root, exist_ok=True)

    # Prepare data if zip exists
    if args.zip_path and os.path.exists(args.zip_path):
        print(f"\n📦 Preparing dataset from {args.zip_path}...")
        prepare_sdnet2018(args.zip_path, args.data_root, args.max_samples)

    # Generate manifests
    print("\n📋 Generating data manifests...")
    generate_manifests(args.data_root, args.output_root)

    # Load data
    print("\n📂 Loading datasets...")
    generators = create_data_generators(args.data_root, args.batch_size)

    if "train" not in generators or "validation" not in generators:
        print("❌ ERROR: train and validation directories are required")
        sys.exit(1)

    print(f"  Train:      {generators['train'].samples} images")
    print(f"  Validation: {generators['validation'].samples} images")
    if "test" in generators:
        print(f"  Test:       {generators['test'].samples} images")

    # Build and train main model
    print("\n🏗️  Building MobileNetV2 model...")
    model, base_model = build_mobilenetv2()
    model.summary()

    model, history_head, history_ft = train(model, base_model, generators, args, args.output_root)

    # Save metadata
    save_metadata(args.output_root, args)

    # Evaluate
    if "test" in generators:
        print("\n📊 Evaluating on test set...")
        metrics, y_prob, y_true, y_pred, threshold = evaluate_model(
            model, generators, args, args.output_root
        )

        # Grad-CAM examples
        print("\n🔍 Generating Grad-CAM examples...")
        generate_correct_gradcam(model, generators, threshold, args.output_root)
        generate_error_case(model, generators, threshold, args.output_root)

        # Hard-negative evaluation
        print("\n⚠️  Evaluating hard negatives...")
        evaluate_hard_negatives(model, args.data_root, args.output_root, threshold)
    else:
        print("[WARN] No test set found — skipping test evaluation")

    # Baseline comparison
    if args.train_baseline and "test" in generators:
        baseline_metrics = train_baseline(generators, args, args.output_root)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Artifacts saved to: {args.output_root}")
    print("\nFiles produced:")
    for root, dirs, files in os.walk(args.output_root):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), args.output_root)
            print(f"  {rel}")


if __name__ == "__main__":
    main()
