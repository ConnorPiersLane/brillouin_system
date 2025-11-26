import cv2
import numpy as np

from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert any image (binary, float, int) to uint8 format in range [0, 255].

    - Binary {0,1} images become {0,255}.
    - Float or other numeric types are normalized to 0–255.
    - uint8 images are returned unchanged.
    """
    if image is None:
        raise ValueError("Input image is None")

    if image.dtype == np.uint8:
        return image

    # Handle binary masks (exactly {0,1})
    unique_vals = np.unique(image)
    if np.array_equal(unique_vals, [0, 1]):
        return (image * 255).astype(np.uint8)

    # Normalize numeric range to 0–255
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image_norm.astype(np.uint8)




def make_black_image(max_size=(512, 512), channels=3) -> np.ndarray:
    """
    Create blank (black) images for left/right cameras.
    """
    width, height = max_size
    shape = (height, width, channels) if channels > 1 else (height, width)
    return np.zeros(shape, dtype=np.uint8)


def scale_image(image, max_size=(512, 512)):
    """
    Scale binary, grayscale, or RGB/BGR image while preserving aspect ratio.

    Automatically chooses interpolation for correct visual output.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    max_size : tuple(int, int)
        Maximum width, height.

    Returns
    -------
    np.ndarray
        Scaled image.
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size

    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Choose interpolation adaptively
    if scale < 1.0:  # downscale
        interp = cv2.INTER_AREA
    else:             # upscale
        interp = cv2.INTER_CUBIC

    # Special case: binary masks
    if np.array_equal(np.unique(image), [0, 1]) or np.array_equal(np.unique(image), [0, 255]):
        interp = cv2.INTER_NEAREST

    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    If image is 2D (grayscale/binary), convert to RGB uint8.
    If already 3-channel, return unchanged.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Ensure uint8 & handle {0,1} binaries robustly
    if image.dtype != np.uint8:
        uniq = np.unique(image)
        if image.dtype == np.bool_ or np.array_equal(uniq, [0, 1]):
            image = (image.astype(np.uint8) * 255)
        else:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim == 3 and image.shape[2] == 3:
        return image  # already color (assume RGB at this point in your pipeline)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def get_opencv_ellipse(ellipse: Ellipse2D):
    """
    Convert an Ellipse2D object to an OpenCV-compatible ellipse tuple.

    Returns
    -------
    tuple | None
        ((cx, cy), (major, minor), angle_deg) if valid,
        otherwise None if any field is invalid or missing.
    """
    # Basic None check
    if ellipse is None:
        return None

    # Check that required fields exist
    attrs = [ellipse.cx, ellipse.cy, ellipse.major, ellipse.minor, ellipse.angle_deg]
    if any(v is None for v in attrs):
        return None

    # Check numerical validity
    if not all(np.isfinite(v) for v in attrs):
        return None

    # Ensure axes are positive and non-zero
    if ellipse.major <= 0 or ellipse.minor <= 0:
        return None

    # Construct OpenCV-style ellipse tuple
    return (
        (float(ellipse.cx), float(ellipse.cy)),
        (float(ellipse.major), float(ellipse.minor)),
        float(ellipse.angle_deg)
    )


def draw_ellipse_rgb(img_rgb, ellipse, color_rgb=(0, 255, 0), thickness=1):
    """Draw ellipse on an RGB image using RGB color input."""
    opencv_ellipse = get_opencv_ellipse(ellipse)


    if opencv_ellipse is None:
        return img_rgb
    else:
        bgr_color = tuple(reversed(color_rgb))  # convert to BGR order
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.ellipse(bgr, opencv_ellipse, bgr_color, thickness)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


