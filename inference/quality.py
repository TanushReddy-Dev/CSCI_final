"""
ConcreteGuard — Image Quality Checks
=====================================

Implements quality validation for uploaded images before inference.
Checks: resolution, blur, brightness, file type, size, and color mode.
"""

import io
import numpy as np
from PIL import Image

# Thresholds
MIN_RESOLUTION = 100           # Minimum width/height in pixels
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
BLUR_THRESHOLD = 50.0          # Laplacian variance below this = blurry
BRIGHTNESS_LOW = 30            # Mean pixel value below this = too dark
BRIGHTNESS_HIGH = 240          # Mean pixel value above this = too bright
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def check_file_type(filename: str) -> dict:
    """Validate file extension."""
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "pass": False,
            "check": "file_type",
            "message": f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        }
    return {"pass": True, "check": "file_type", "message": "File type is acceptable."}


def check_file_size(file_bytes: bytes) -> dict:
    """Validate file size."""
    size = len(file_bytes)
    if size > MAX_FILE_SIZE:
        mb = size / (1024 * 1024)
        return {
            "pass": False,
            "check": "file_size",
            "message": f"File is {mb:.1f} MB. Maximum allowed is {MAX_FILE_SIZE / (1024*1024):.0f} MB.",
        }
    return {"pass": True, "check": "file_size", "message": "File size is acceptable."}


def check_resolution(image: Image.Image) -> dict:
    """Check minimum resolution."""
    w, h = image.size
    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        return {
            "pass": False,
            "check": "resolution",
            "message": f"Image resolution ({w}×{h}) is below minimum ({MIN_RESOLUTION}×{MIN_RESOLUTION}).",
        }
    return {"pass": True, "check": "resolution", "message": f"Resolution ({w}×{h}) is acceptable."}


def check_blur(image: Image.Image) -> dict:
    """Check image sharpness using Laplacian variance."""
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float64)

    # Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    # Simple convolution (no scipy dependency)
    from numpy.lib.stride_tricks import sliding_window_view
    h, w = arr.shape
    if h < 3 or w < 3:
        return {"pass": True, "check": "blur", "message": "Image too small for blur check."}

    windows = sliding_window_view(arr, (3, 3))
    laplacian = np.sum(windows * kernel, axis=(-2, -1))
    variance = np.var(laplacian)

    if variance < BLUR_THRESHOLD:
        return {
            "pass": False,
            "check": "blur",
            "message": f"Image appears blurry (sharpness: {variance:.1f}, threshold: {BLUR_THRESHOLD}).",
            "score": float(variance),
        }
    return {
        "pass": True,
        "check": "blur",
        "message": f"Image sharpness is acceptable ({variance:.1f}).",
        "score": float(variance),
    }


def check_brightness(image: Image.Image) -> dict:
    """Check if image is too dark or too bright."""
    gray = image.convert("L")
    mean_val = np.mean(np.array(gray))

    if mean_val < BRIGHTNESS_LOW:
        return {
            "pass": False,
            "check": "brightness",
            "message": f"Image is too dark (mean brightness: {mean_val:.0f}, minimum: {BRIGHTNESS_LOW}).",
            "score": float(mean_val),
        }
    if mean_val > BRIGHTNESS_HIGH:
        return {
            "pass": False,
            "check": "brightness",
            "message": f"Image is too bright/overexposed (mean brightness: {mean_val:.0f}, maximum: {BRIGHTNESS_HIGH}).",
            "score": float(mean_val),
        }
    return {
        "pass": True,
        "check": "brightness",
        "message": f"Image brightness is acceptable ({mean_val:.0f}).",
        "score": float(mean_val),
    }


def normalize_color_mode(image: Image.Image) -> Image.Image:
    """Convert grayscale/RGBA images to RGB."""
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    if image.mode == "L":
        return image.convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def run_all_checks(file_bytes: bytes, filename: str) -> dict:
    """
    Run all quality checks on an uploaded image.
    
    Returns:
        {
            "status": "acceptable" | "warning" | "insufficient",
            "message": str,
            "checks": [dict],
            "image": PIL.Image or None,
        }
    """
    results = []

    # File type check
    results.append(check_file_type(filename))
    if not results[-1]["pass"]:
        return {
            "status": "insufficient",
            "message": results[-1]["message"],
            "checks": results,
            "image": None,
        }

    # File size check
    results.append(check_file_size(file_bytes))
    if not results[-1]["pass"]:
        return {
            "status": "insufficient",
            "message": results[-1]["message"],
            "checks": results,
            "image": None,
        }

    # Try to open image
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # Force load to catch corrupt files
    except Exception as e:
        return {
            "status": "insufficient",
            "message": f"Cannot open image: {e}",
            "checks": results,
            "image": None,
        }

    # Resolution
    results.append(check_resolution(image))

    # Blur
    results.append(check_blur(image))

    # Brightness
    results.append(check_brightness(image))

    # Normalize color mode
    image = normalize_color_mode(image)

    # Aggregate
    failures = [r for r in results if not r["pass"]]
    warnings = [r for r in results if not r["pass"] and r["check"] in ("blur", "brightness")]

    if any(r["check"] in ("file_type", "file_size", "resolution") and not r["pass"] for r in results):
        status = "insufficient"
        message = "; ".join(r["message"] for r in failures)
    elif warnings:
        status = "warning"
        message = "; ".join(r["message"] for r in warnings) + " Results may be less reliable."
    else:
        status = "acceptable"
        message = "Image quality is acceptable for visual screening."

    return {
        "status": status,
        "message": message,
        "checks": results,
        "image": image,
    }
