"""
Image preprocessing utilities for pharmaceutical package and sheet recognition.
"""

import cv2
import numpy as np
from loguru import logger

def resize_with_aspect_ratio(image, max_size=1000):
    """
    Resize image while preserving aspect ratio.
    
    Args:
        image (numpy.ndarray): Input image
        max_size (int): Maximum dimension (width or height)
        
    Returns:
        numpy.ndarray: Resized image
    """
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    # Return original image if already smaller than max_size
    if max(height, width) <= max_size:
        return image
    
    # Calculate new dimensions
    if height > width:
        new_height = max_size
        new_width = int(width * (new_height / height))
    else:
        new_width = max_size
        new_height = int(height * (new_width / width))
    
    # Resize image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """
    Normalize image for consistent processing.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    if image is None:
        return None
    
    # Ensure proper color format
    if len(image.shape) == 2:
        # Convert grayscale to BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # Convert BGRA to BGR
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 3:
        # Already BGR
        return image.copy()
    else:
        logger.warning(f"Unexpected image format with shape {image.shape}")
        return image

def enhance_contrast(image):
    """
    Enhance image contrast for better feature detection.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    if image is None:
        return None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Return enhanced image in original format
    if len(image.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        return enhanced

def remove_noise(image, strength='medium'):
    """
    Remove noise from image.
    
    Args:
        image (numpy.ndarray): Input image
        strength (str): Noise removal strength ('light', 'medium', 'strong')
        
    Returns:
        numpy.ndarray: Noise-reduced image
    """
    if image is None:
        return None
    
    # Convert to grayscale if needed
    is_color = len(image.shape) == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply noise removal based on strength
    if strength == 'light':
        # Gaussian blur
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    elif strength == 'medium':
        # Bilateral filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    elif strength == 'strong':
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    else:
        logger.warning(f"Unknown noise removal strength: {strength}")
        denoised = gray
    
    # Return denoised image in original format
    if is_color:
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    else:
        return denoised

def correct_perspective(image):
    """
    Correct perspective distortion in image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Perspective-corrected image
    """
    if image is None:
        return None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have a quadrilateral, correct perspective
    if len(approx) == 4:
        # Sort points by x+y value (top-left, top-right, bottom-right, bottom-left)
        points = approx.reshape(4, 2).astype(np.float32)
        points = points[np.argsort(np.sum(points, axis=1))]
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Define destination points (rectangle)
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(points, dst_points)
        
        # Apply perspective transformation
        return cv2.warpPerspective(image, M, (width, height))
    
    return image

def handle_reflective_surface(image):
    """
    Process image to reduce problems with reflective surfaces.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Processed image
    """
    if image is None:
        return None
    
    # Convert to grayscale if needed
    is_color = len(image.shape) == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Invert back
    processed = cv2.bitwise_not(binary)
    
    # Return processed image in original format
    if is_color:
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        return processed

def detect_and_crop_document(image, padding=10):
    """
    Detect and crop document from image.
    
    Args:
        image (numpy.ndarray): Input image
        padding (int): Padding around detected document
        
    Returns:
        numpy.ndarray: Cropped document image
    """
    if image is None:
        return None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Apply dilation to connect edge gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Apply padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Crop image
    return image[y:y+h, x:x+w]

def preprocess_for_text_detection(image):
    """
    Preprocess image for optimal text detection.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    if image is None:
        return None
    
    # Resize large images for consistent processing
    image = resize_with_aspect_ratio(image, max_size=1500)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    return enhanced

def sharpen_image(image):
    """
    Sharpen image for better text clarity.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Sharpened image
    """
    if image is None:
        return None
    
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    
    # Apply kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened

def detect_handwritten_regions(image):
    """
    Detect regions with handwritten text in an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        list: List of handwritten region bounding boxes [x, y, w, h]
    """
    if image is None:
        return []
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive threshold to enhance handwriting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
    )
    
    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    handwritten_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Handwritten regions typically have specific aspect ratios
            if 0.2 < aspect_ratio < 10:
                handwritten_regions.append([x, y, w, h])
    
    return handwritten_regions

def detect_stickers(image):
    """
    Detect stickers on pharmaceutical packages.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        list: List of sticker region bounding boxes [x, y, w, h]
    """
    if image is None:
        return []
    
    # Convert to HSV color space
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        # If grayscale, can't detect stickers by color
        return []
    
    # Define color ranges for common sticker colors
    color_ranges = [
        # Yellow
        (np.array([20, 100, 100]), np.array([40, 255, 255])),
        # Green
        (np.array([40, 100, 100]), np.array([80, 255, 255])),
        # Red (wraps around in HSV, so two ranges)
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255])),
        # White/light
        (np.array([0, 0, 200]), np.array([180, 30, 255]))
    ]
    
    # Combine masks for all color ranges
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    sticker_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold for stickers
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Stickers typically have aspect ratios close to 1
            if 0.5 < aspect_ratio < 2.0:
                sticker_regions.append([x, y, w, h])
    
    return sticker_regions

def detect_barcodes(image):
    """
    Detect barcodes in an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        list: List of barcode region bounding boxes [x, y, w, h]
    """
    if image is None:
        return []
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Scharr gradient (more sensitive than Sobel)
    gradX = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    gradY = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    
    # Subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # Blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Perform multiple dilations and erosions
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    barcode_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold for barcodes
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Barcodes typically have wider aspect ratios
            if aspect_ratio > 2.0:
                barcode_regions.append([x, y, w, h])
    
    return barcode_regions
