"""
ConcreteGuard — Grad-CAM Visualization
=======================================

Generates Grad-CAM heatmaps for model predictions.
Used by the inference API to provide visual evidence of what the model focused on.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import base64

IMG_SIZE = (224, 224)


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model (handles Sequential + nested)."""
    for layer in reversed(model.layers):
        # Check nested models (e.g., MobileNetV2 base inside Sequential)
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if "out_relu" in sublayer.name.lower() or "conv" in sublayer.name.lower() or isinstance(sublayer, (keras.layers.Conv2D, tf.keras.layers.Conv2D)):
                    return sublayer.name, layer
            continue
        if "conv" in layer.name.lower() or isinstance(layer, (keras.layers.Conv2D, tf.keras.layers.Conv2D)):
            return layer.name, None
    return None, None


def generate_gradcam(model, img_array, layer_name=None):
    """
    Generate Grad-CAM heatmap.
    
    Args:
        model: Keras model
        img_array: (1, 224, 224, 3) float32 array scaled to [0, 1]
        layer_name: Target conv layer name. Auto-detected if None.
    
    Returns:
        heatmap: (224, 224) numpy array in [0, 1]
    """
    if layer_name is None:
        layer_name, parent = find_last_conv_layer(model)
        if layer_name is None:
            return np.zeros(IMG_SIZE, dtype=np.float32)

    # Build gradient model with Keras 3 & nested MobileNetV2 support
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
                    if "out_relu" in l.name.lower() or "conv" in l.name.lower() or isinstance(l, (keras.layers.Conv2D, tf.keras.layers.Conv2D)):
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
            # Flat model
            target_layer = None
            if layer_name:
                try:
                    target_layer = model.get_layer(layer_name)
                except Exception:
                    pass
            if target_layer is None:
                for l in reversed(model.layers):
                    if isinstance(l, (keras.layers.Conv2D, tf.keras.layers.Conv2D)) or "conv" in l.name.lower():
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
            return np.zeros(IMG_SIZE, dtype=np.float32)

        # Weighted activation map
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(
            tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs),
            axis=-1
        )
        cam = tf.nn.relu(cam).numpy()[0]

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to image size
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(IMG_SIZE, Image.BILINEAR)
        ).astype(np.float32) / 255.0

        return cam_resized
    except Exception as e:
        print(f"[WARN] Grad-CAM calculation failed: {e}")
        return np.zeros(IMG_SIZE, dtype=np.float32)


def create_heatmap_overlay(original_img, heatmap, alpha=0.4, colormap="jet"):
    """
    Create a Grad-CAM overlay image.
    
    Args:
        original_img: (224, 224, 3) numpy array in [0, 1]
        heatmap: (224, 224) numpy array in [0, 1]
        alpha: Overlay opacity
        colormap: Matplotlib colormap name
    
    Returns:
        overlay: PIL Image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Blend
    overlay = (1 - alpha) * original_img + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    return Image.fromarray(overlay)


def gradcam_to_base64(original_img, heatmap, alpha=0.4):
    """Generate Grad-CAM overlay and return as base64-encoded PNG."""
    overlay = create_heatmap_overlay(original_img, heatmap, alpha)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_gradcam_comparison(original_img, heatmap, save_path, title="Grad-CAM Analysis"):
    """Save a side-by-side comparison: original | heatmap | overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay = create_heatmap_overlay(original_img, heatmap)
    overlay_arr = np.array(overlay).astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_arr)
    axes[2].set_title("Evidence Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
