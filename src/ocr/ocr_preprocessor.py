"""
OCR preprocessing module for enhancing image quality before text recognition.
"""

import cv2
import numpy as np
from loguru import logger

class OCRPreprocessor:
    """
    Provides specialized preprocessing techniques for pharmaceutical text.
    Implements different preprocessing strategies for packages and information sheets.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize OCR preprocessor.
        
        Args:
            config_path (str): Path to configuration file
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.special_cases = config['special_cases']
        
        # Configure preprocessing options
        self.handle_reflective = self.special_cases.get('handle_reflective_surfaces', True)
        self.handle_perspective = self.special_cases.get('handle_perspective_distortion', True)
        
        logger.info("OCR preprocessor initialized")
    
    def preprocess(self, image, is_package=True, preprocessing_options=None):
        """
        Preprocess image for OCR.
        
        Args:
            image (numpy.ndarray): Input image
            is_package (bool): True if the image is from a package, False for sheet
            preprocessing_options (dict, optional): Custom preprocessing options
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for preprocessing")
            return image
        
        # Default options based on document type
        if preprocessing_options is None:
            if is_package:
                preprocessing_options = {
                    'enhance_contrast': True,
                    'remove_noise': True,
                    'handle_reflective': self.handle_reflective,
                    'handle_perspective': self.handle_perspective,
                    'sharpen': False,
                    'denoise': True
                }
            else:
                preprocessing_options = {
                    'enhance_contrast': False,
                    'remove_noise': True,
                    'handle_reflective': False,
                    'handle_perspective': False,
                    'sharpen': True,
                    'denoise': False
                }
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing techniques based on options
        processed = gray.copy()
        
        # Enhanced contrast
        if preprocessing_options.get('enhance_contrast', False):
            processed = self._enhance_contrast(processed)
        
        # Remove noise
        if preprocessing_options.get('remove_noise', False):
            processed = self._remove_noise(processed)
        
        # Handle reflective surfaces
        if preprocessing_options.get('handle_reflective', False):
            processed = self._handle_reflective_surface(processed)
        
        # Handle perspective distortion
        if preprocessing_options.get('handle_perspective', False):
            processed = self._correct_perspective(processed)
        
        # Sharpen
        if preprocessing_options.get('sharpen', False):
            processed = self._sharpen_image(processed)
        
        # Denoise
        if preprocessing_options.get('denoise', False):
            processed = self._denoise_image(processed)
        
        # Return processed image
        # Convert back to original format (color or grayscale)
        if len(image.shape) == 3:
            return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            return processed
    
    def _enhance_contrast(self, image):
        """
        Enhance image contrast for better text visibility.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Contrast-enhanced image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _remove_noise(self, image):
        """
        Remove noise from image.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Noise-reduced image
        """
        # Apply bilateral filter to reduce noise while preserving edges
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _handle_reflective_surface(self, image):
        """
        Process image to reduce problems with reflective surfaces.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Processed image
        """
        # Apply adaptive thresholding to handle reflective surfaces
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert back
        return cv2.bitwise_not(binary)
    
    def _correct_perspective(self, image):
        """
        Correct perspective distortion in image.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Perspective-corrected image
        """
        # Find contours
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (assuming it's the document)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get approximate polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have a quadrilateral, correct perspective
        if len(approx) == 4:
            # Sort points by x+y value (top-left, top-right, bottom-right, bottom-left)
            points = approx.reshape(4, 2).astype(np.float32)
            points = points[np.argsort(np.sum(points, axis=1))]
            
            # Define destination points (rectangle)
            width = max(
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[3] - points[2])
            )
            height = max(
                np.linalg.norm(points[2] - points[1]),
                np.linalg.norm(points[3] - points[0])
            )
            
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(points, dst_points)
            
            # Apply perspective transformation
            return cv2.warpPerspective(image, M, (int(width), int(height)))
        
        return image
    
    def _sharpen_image(self, image):
        """
        Sharpen image for better text clarity.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Sharpened image
        """
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        
        # Apply kernel
        return cv2.filter2D(image, -1, kernel)
    
    def _denoise_image(self, image):
        """
        Remove noise from image using more aggressive methods.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Denoised image
        """
        # Apply non-local means denoising
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def preprocess_package(self, image):
        """
        Apply package-specific preprocessing.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Define package-specific options
        options = {
            'enhance_contrast': True,
            'remove_noise': True,
            'handle_reflective': True,
            'handle_perspective': True,
            'sharpen': False,
            'denoise': True
        }
        
        return self.preprocess(image, is_package=True, preprocessing_options=options)
    
    def preprocess_sheet(self, image):
        """
        Apply sheet-specific preprocessing.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Define sheet-specific options
        options = {
            'enhance_contrast': False,
            'remove_noise': True,
            'handle_reflective': False,
            'handle_perspective': True,
            'sharpen': True,
            'denoise': False
        }
        
        return self.preprocess(image, is_package=False, preprocessing_options=options)
    
    def preprocess_arabic_text(self, image):
        """
        Apply specialized preprocessing for Arabic text.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE with higher clip limit for Arabic text
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Apply thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to connect disconnected parts of Arabic characters
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Return processed image in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        else:
            return morph
    
    def preprocess_handwritten_regions(self, image):
        """
        Apply specialized preprocessing for handwritten annotations.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive threshold to enhance handwriting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
        )
        
        # Dilate to connect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Invert back
        processed = cv2.bitwise_not(dilated)
        
        # Return processed image in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            return processed
