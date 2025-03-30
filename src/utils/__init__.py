"""
Utility modules for pharmaceutical package and sheet recognition.
"""

from .image_preprocessor import (
    resize_with_aspect_ratio,
    normalize_image,
    enhance_contrast,
    remove_noise,
    correct_perspective,
    handle_reflective_surface,
    detect_and_crop_document,
    preprocess_for_text_detection,
    sharpen_image,
    detect_handwritten_regions,
    detect_stickers,
    detect_barcodes,
)

from .performance_monitor import PerformanceTimer, ComponentProfiler
from .cuda_utils import CUDAUtils
from .validation import ValidationUtils
