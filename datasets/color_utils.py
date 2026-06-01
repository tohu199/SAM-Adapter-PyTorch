"""RGB color mask utilities for 3-class segmentation.

Class mapping:
  0 - black  (0, 0, 0)       background
  1 - yellow (255, 255, 0)
  2 - red    (255, 0, 0)
"""

import numpy as np
from PIL import Image

# index = class_id
PALETTE = np.array([
    [0, 0, 0],
    [255, 255, 0],
    [255, 0, 0],
], dtype=np.int32)

NUM_CLASSES = 3

LABEL_TO_RGB = PALETTE.astype(np.uint8)


def rgb_to_label(rgb: np.ndarray) -> np.ndarray:
    """Map RGB image to class indices via nearest palette color.

    Args:
        rgb: uint8 array of shape (H, W, 3)

    Returns:
        int64 array of shape (H, W) with values in {0, 1, 2}
    """
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError(f'Expected 3 channels, got {c}')
    flat = rgb.reshape(-1, 3).astype(np.int32)
    dist = ((flat[:, None, :] - PALETTE[None, :, :]) ** 2).sum(axis=-1)
    return dist.argmin(axis=1).reshape(h, w).astype(np.int64)


def rgb_pil_to_label(mask: Image.Image) -> np.ndarray:
    return rgb_to_label(np.array(mask.convert('RGB')))


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """Convert class map (H, W) to RGB uint8 (H, W, 3)."""
    return LABEL_TO_RGB[label.astype(np.int64)]
