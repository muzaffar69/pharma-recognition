"""
OCR engine for pharmaceutical package and sheet recognition.
Uses PaddleOCR with optimizations for speed and accuracy.
"""

import os
import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR, draw_ocr
from loguru import logger
import time

from ..utils.performance_monitor import PerformanceTimer
from ..config.ocr_config import OCRConfig

class OCREngine:
    """
    Highly optimized OCR engine for pharmaceutical text detection and recognition.
    Provides specialized processing for both package and sheet recognition.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize the OCR engine.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Initialize OCR configuration
        self.config = OCRConfig(config_path)
        
        # Initialize standard OCR engine
        self.standard_ocr = self._initialize_ocr(self.config.get_base_ocr_params())
        
        # Initialize specialized OCR engines
        self.package_ocr = self._initialize_ocr(self.config.get_package_ocr_params())
        self.sheet_ocr = self._initialize_ocr(self.config.get_sheet_ocr_params())
        self.fast_ocr = self._initialize_ocr(self.config.get_fast_mode_params())
        
        # Performance timer
        self.timer = PerformanceTimer()
        
        logger.info("OCR engine initialized with specialized processors")
        
        # Warm up the OCR engines to reduce first inference latency
        if self.config.use_gpu_warmup:
            self._warm_up()
    
    def _initialize_ocr(self, params):
        """
        Initialize PaddleOCR with specific parameters.
        
        Args:
            params (dict): OCR parameters
            
        Returns:
            PaddleOCR: Initialized OCR engine
        """
        try:
            # Make a copy of params to avoid modifying the original
            ocr_params = params.copy()
            
            # Initialize PaddleOCR
            ocr = PaddleOCR(**ocr_params)
            
            return ocr
        except Exception as e:
            logger.error(f"Error initializing OCR engine: {e}")
            # Fall back to minimal configuration
            fallback_params = {
                'use_angle_cls': False,
                'lang': 'en'
            }
            return PaddleOCR(**fallback_params)
    
    def _warm_up(self):
        """Warm up OCR engines with a dummy inference to reduce first inference latency."""
        logger.info("Warming up OCR engines...")
        
        # Create dummy image
        dummy_image = np.ones((100, 300), dtype=np.uint8) * 255
        cv2.putText(dummy_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Warm up OCR engines
        try:
            with self.timer.measure("ocr_warmup"):
                self.standard_ocr.ocr(dummy_image, cls=False)
                self.package_ocr.ocr(dummy_image, cls=False)
                self.sheet_ocr.ocr(dummy_image, cls=False)
                self.fast_ocr.ocr(dummy_image, cls=False)
            
            logger.info(f"OCR engines warmed up in {self.timer.get_elapsed('ocr_warmup'):.2f}ms")
        except Exception as e:
            logger.warning(f"OCR warmup failed: {e}")
    
    def recognize_text(self, image, is_package=True, fast_mode=False, config_override=None):
        """
        Perform OCR on an image.
        
        Args:
            image (numpy.ndarray): Input image
            is_package (bool): True if the image is from a package, False for sheet
            fast_mode (bool): Use fast mode for time-critical operations
            config_override (dict, optional): Override OCR parameters
            
        Returns:
            list: List of detected text results
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to OCR")
            return []
        
        # Select appropriate OCR engine
        if fast_mode:
            ocr_engine = self.fast_ocr
        elif is_package:
            ocr_engine = self.package_ocr
        else:
            ocr_engine = self.sheet_ocr
        
        # Apply custom configuration if provided
        if config_override:
            # Create a temporary OCR engine with overridden configuration
            base_params = self.config.get_package_ocr_params() if is_package else self.config.get_sheet_ocr_params()
            params = {**base_params, **config_override}
            ocr_engine = self._initialize_ocr(params)
        
        # Perform OCR
        self.timer.start("ocr_recognition")
        
        try:
            # Get OCR results
            results = ocr_engine.ocr(image, cls=False)
            
            # Process results into a consistent format
            processed_results = self._process_ocr_results(results)
            
            elapsed_ms = self.timer.stop("ocr_recognition")
            logger.debug(f"OCR completed in {elapsed_ms:.2f}ms, detected {len(processed_results)} text regions")
            
            return processed_results
        except Exception as e:
            elapsed_ms = self.timer.stop("ocr_recognition")
            logger.error(f"OCR failed after {elapsed_ms:.2f}ms: {e}")
            return []
    
    def _process_ocr_results(self, results):
        """
        Process OCR results into a consistent format.
        
        Args:
            results (list): PaddleOCR results
            
        Returns:
            list: Processed results as list of dicts
        """
        processed_results = []
        
        # Handle different result formats from different PaddleOCR versions
        if not results:
            return []
        
        # Handle results from newer PaddleOCR versions (list of list of tuples)
        if isinstance(results, list) and results and isinstance(results[0], list):
            for result in results:
                for text_region in result:
                    if isinstance(text_region, tuple) and len(text_region) == 2:
                        box, (text, confidence) = text_region
                        
                        processed_results.append({
                            'box': box,
                            'text': text,
                            'confidence': confidence
                        })
        
        # Handle results from older PaddleOCR versions (list of tuples)
        elif isinstance(results, list) and results and isinstance(results[0], tuple):
            for box, (text, confidence) in results:
                processed_results.append({
                    'box': box,
                    'text': text,
                    'confidence': confidence
                })
        
        return processed_results
    
    def extract_field(self, rois, field_name, is_package=True, min_confidence=0.7):
        """
        Extract a specific field from ROIs.
        
        Args:
            rois (dict): ROIs for each field
            field_name (str): Field name to extract
            is_package (bool): True if processing a package, False for sheet
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            dict: Extracted field data with value and confidence
        """
        if field_name not in rois or not rois[field_name]:
            logger.warning(f"No ROIs available for field {field_name}")
            return {'value': None, 'confidence': 0.0}
        
        # Process ROIs in priority order
        results = []
        
        for roi_index, roi_data in enumerate(rois[field_name]):
            roi_image = roi_data['image']
            ocr_config = roi_data.get('ocr_config', {})
            
            # Recognize text in ROI
            text_results = self.recognize_text(roi_image, is_package, config_override=ocr_config)
            
            for text_result in text_results:
                if text_result['confidence'] >= min_confidence:
                    results.append({
                        'text': text_result['text'],
                        'confidence': text_result['confidence'],
                        'roi_index': roi_index,
                        'priority': roi_data.get('priority', 0)
                    })
        
        # If no results found
        if not results:
            return {'value': None, 'confidence': 0.0}
        
        # Sort by confidence and priority
        results.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)
        
        # Return best result
        best_result = results[0]
        return {
            'value': best_result['text'],
            'confidence': best_result['confidence'],
            'source_roi': best_result['roi_index']
        }
    
    def extract_fields(self, rois, is_package=True, min_confidence=0.7):
        """
        Extract all fields from ROIs.
        
        Args:
            rois (dict): ROIs for each field
            is_package (bool): True if processing a package, False for sheet
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            dict: Dictionary of extracted fields with values and confidences
        """
        if not rois:
            logger.warning("No ROIs provided for field extraction")
            return {}
        
        # Extract each field
        results = {}
        
        for field_name in rois.keys():
            field_result = self.extract_field(rois, field_name, is_package, min_confidence)
            results[field_name] = field_result
        
        return results
    
    def detect_language(self, image):
        """
        Detect language in an image to optimize OCR parameters.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Detected languages with confidence scores
        """
        # Simplified language detection using features of detected text
        self.timer.start("language_detection")
        
        # Use standard OCR to get text
        try:
            results = self.standard_ocr.ocr(image, cls=False)
            processed_results = self._process_ocr_results(results)
        except Exception as e:
            logger.error(f"Language detection OCR failed: {e}")
            self.timer.stop("language_detection")
            return {'en': 1.0}  # Default to English
        
        if not processed_results:
            self.timer.stop("language_detection")
            return {'en': 1.0}  # Default to English
        
        # Analyze text for language markers
        text = ' '.join(r['text'] for r in processed_results)
        
        # Simple character set analysis
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        digit_chars = sum(1 for c in text if c.isdigit())
        
        total_chars = len(text) - text.count(' ')
        if total_chars == 0:
            self.timer.stop("language_detection")
            return {'en': 1.0}  # Default to English
        
        # Calculate language probabilities
        arabic_ratio = arabic_chars / total_chars
        english_ratio = english_chars / total_chars
        
        languages = {}
        
        if arabic_ratio > 0.3:
            languages['ar'] = arabic_ratio
        
        if english_ratio > 0.3:
            languages['en'] = english_ratio
        
        # Ensure we return at least one language
        if not languages:
            languages['en'] = 1.0
        
        # Normalize probabilities
        total_prob = sum(languages.values())
        for lang in languages:
            languages[lang] /= total_prob
        
        self.timer.stop("language_detection")
        return languages
    
    def visualize_results(self, image, results):
        """
        Visualize OCR results on an image for debugging.
        
        Args:
            image (numpy.ndarray): Input image
            results (list): OCR results
            
        Returns:
            numpy.ndarray: Image with visualized results
        """
        if not results:
            return image.copy()
        
        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = image.copy()
        
        # Extract boxes and texts
        boxes = [result['box'] for result in results]
        texts = [result['text'] for result in results]
        scores = [result['confidence'] for result in results]
        
        # Draw results
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for box, text, score in zip(boxes, texts, scores):
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_image, [box], True, (0, 255, 0), 2)
            
            # Draw text
            min_x = min(point[0] for point in box)
            min_y = min(point[1] for point in box)
            cv2.putText(display_image, f"{text} ({score:.2f})", (min_x, min_y - 10),
                       font, 0.5, (0, 0, 255), 1)
        
        return display_image
    
    def optimize_for_orientation(self, image):
        """
        Optimize OCR by detecting and correcting text orientation.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Orientation-corrected image
        """
        self.timer.start("orientation_detection")
        
        try:
            # Use angle classifier to detect orientation
            angle_results = self.standard_ocr.ocr(image, det=False, rec=False, cls=True)
            
            if angle_results and isinstance(angle_results, tuple) and len(angle_results) == 2:
                angle, probability = angle_results
                
                # Only rotate if probability is high enough
                if probability > 0.8 and angle != 0:
                    # Convert angle to rotation matrix parameters
                    if angle == 180:
                        corrected = cv2.rotate(image, cv2.ROTATE_180)
                    elif angle == 90:
                        corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif angle == 270:
                        corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        corrected = image
                    
                    self.timer.stop("orientation_detection")
                    return corrected
        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
        
        self.timer.stop("orientation_detection")
        return image
