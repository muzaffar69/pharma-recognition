"""
Specialized OCR processor for pharmaceutical packages.
"""

import cv2
import numpy as np
from loguru import logger

from .ocr_engine import OCREngine
from .ocr_preprocessor import OCRPreprocessor
from .ocr_postprocessor import OCRPostprocessor
from ..utils.performance_monitor import PerformanceTimer

class PackageProcessor:
    """
    Specialized OCR processor for pharmaceutical packages.
    Handles package-specific challenges such as:
    - Reflective surfaces
    - Curved surfaces
    - Perspective distortion
    - Text at various orientations
    - Missing tablets
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize package processor.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        
        # Initialize components
        self.ocr_engine = OCREngine(config_path)
        self.preprocessor = OCRPreprocessor(config_path)
        self.postprocessor = OCRPostprocessor(config_path)
        
        # Performance timer
        self.timer = PerformanceTimer()
        
        logger.info("Package processor initialized")
    
    def process(self, image, rois):
        """
        Process package image with ROIs to extract information.
        
        Args:
            image (numpy.ndarray): Package image
            rois (dict): ROIs for each field
            
        Returns:
            dict: Extracted field data
        """
        if image is None or rois is None:
            logger.warning("Invalid inputs for package processing")
            return {}
        
        self.timer.start("package_processing")
        
        # Process each field
        results = {}
        
        for field_name, field_rois in rois.items():
            field_results = self._process_field(image, field_rois, field_name)
            results[field_name] = field_results
        
        elapsed_ms = self.timer.stop("package_processing")
        logger.debug(f"Package processing completed in {elapsed_ms:.2f}ms")
        
        return results
    
    def _process_field(self, image, field_rois, field_name):
        """
        Process a specific field from ROIs.
        
        Args:
            image (numpy.ndarray): Package image
            field_rois (list): ROIs for the field
            field_name (str): Field name
            
        Returns:
            dict: Extracted field data
        """
        # Skip empty ROIs
        if not field_rois:
            return {'value': None, 'confidence': 0.0}
        
        # Process each ROI and collect results
        all_results = []
        
        for roi_index, roi_data in enumerate(field_rois):
            roi_image = roi_data['image']
            ocr_config = roi_data.get('ocr_config', {})
            
            # Skip empty ROI images
            if roi_image is None or roi_image.size == 0:
                continue
            
            # Preprocess ROI
            preprocessed = self.preprocessor.preprocess_package(roi_image)
            
            # Apply specialized preprocessing based on field type
            if field_name == 'dosage':
                # Enhance contrast for dosage values
                preprocessed = self._enhance_dosage_region(preprocessed)
            elif field_name == 'commercial_name':
                # Apply logo/brand name enhancement
                preprocessed = self._enhance_brand_region(preprocessed)
            
            # Apply OCR
            try:
                text_results = self.ocr_engine.recognize_text(preprocessed, is_package=True, config_override=ocr_config)
                
                # For small ROIs, try again with orientation detection if no text found
                if not text_results and max(roi_image.shape) > 100:
                    rotated = self.ocr_engine.optimize_for_orientation(preprocessed)
                    if not np.array_equal(rotated, preprocessed):
                        text_results = self.ocr_engine.recognize_text(rotated, is_package=True, config_override=ocr_config)
                
                # Add roi_index to each result
                for result in text_results:
                    result['roi_index'] = roi_index
                    result['priority'] = roi_data.get('priority', 0)
                
                # Apply postprocessing
                processed_results = self.postprocessor.postprocess_results(text_results, field_name)
                
                # Filter by confidence
                filtered_results = self.postprocessor.filter_by_confidence(processed_results, field_name)
                
                all_results.extend(filtered_results)
                
            except Exception as e:
                logger.error(f"Error processing ROI for {field_name}: {e}")
        
        # If no results found
        if not all_results:
            return {'value': None, 'confidence': 0.0}
        
        # Sort by confidence and priority
        all_results.sort(key=lambda x: (x.get('priority', 0), x.get('confidence', 0)), reverse=True)
        
        # Take best result
        best_result = all_results[0]
        
        return {
            'value': best_result.get('text', ''),
            'confidence': best_result.get('confidence', 0.0),
            'source_roi': best_result.get('roi_index', -1)
        }
    
    def _enhance_dosage_region(self, image):
        """
        Apply specialized enhancement for dosage regions.
        
        Args:
            image (numpy.ndarray): ROI image
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding optimized for digit detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )
        
        # Apply morphological operations optimized for digits
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Return in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            return processed
    
    def _enhance_brand_region(self, image):
        """
        Apply specialized enhancement for brand name/logo regions.
        
        Args:
            image (numpy.ndarray): ROI image
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Return in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            return enhanced
    
    def process_curved_text(self, image, rois):
        """
        Process text on curved surfaces (e.g., blister packs).
        
        Args:
            image (numpy.ndarray): Package image
            rois (dict): ROIs for each field
            
        Returns:
            dict: Extracted field data
        """
        # Start with standard processing
        results = self.process(image, rois)
        
        # For fields with low confidence, try curved text detection
        for field_name, field_result in results.items():
            if field_result.get('confidence', 0) < 0.7 and field_name in rois:
                field_rois = rois[field_name]
                
                # Process each ROI with curved text methods
                curved_results = []
                
                for roi_index, roi_data in enumerate(field_rois):
                    roi_image = roi_data['image']
                    
                    # Skip empty ROI images
                    if roi_image is None or roi_image.size == 0:
                        continue
                    
                    # Apply curved text detection
                    curved_text = self._detect_curved_text(roi_image)
                    
                    if curved_text:
                        curved_results.append({
                            'text': curved_text,
                            'confidence': 0.75,  # Default confidence for curved text
                            'roi_index': roi_index,
                            'priority': roi_data.get('priority', 0)
                        })
                
                # If curved text detection found results, use them
                if curved_results:
                    # Sort by priority
                    curved_results.sort(key=lambda x: x.get('priority', 0), reverse=True)
                    
                    # Take best result
                    best_result = curved_results[0]
                    
                    results[field_name] = {
                        'value': best_result.get('text', ''),
                        'confidence': best_result.get('confidence', 0.0),
                        'source_roi': best_result.get('roi_index', -1)
                    }
        
        return results
    
    def _detect_curved_text(self, image):
        """
        Detect text on curved surfaces.
        
        Args:
            image (numpy.ndarray): ROI image
            
        Returns:
            str: Detected text or None
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get all text-like contours
        text_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50 and area < 5000:
                # Filter by aspect ratio (typical for text)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.2 < aspect_ratio < 5.0:
                    text_contours.append(contour)
        
        if not text_contours:
            return None
        
        # Create mask for text regions
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, text_contours, -1, 255, -1)
        
        # Apply mask to original image
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Try OCR on the masked image
        try:
            text_results = self.ocr_engine.recognize_text(masked, is_package=True)
            
            # Combine text results
            text = ' '.join([r.get('text', '') for r in text_results if r.get('confidence', 0) > 0.5])
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"Error detecting curved text: {e}")
            return None
