"""
Specialized ROI handling for pharmaceutical packages.
"""

import cv2
import numpy as np
from loguru import logger

from .roi_mapper import ROIMapper

class PackageROI:
    """
    Handles ROI operations specifically for pharmaceutical packages.
    Provides specialized extraction methods for package-specific challenges:
    - Curved surfaces
    - Reflective materials
    - Perforations and missing tablets
    - Text at various orientations
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize package ROI handler.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.roi_mapper = ROIMapper(config_path)
        
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.special_cases = config['special_cases']
        
        # Package-specific extraction settings
        self.handle_reflective = self.special_cases.get('handle_reflective_surfaces', True)
        self.handle_perspective = self.special_cases.get('handle_perspective_distortion', True)
        
        logger.info("Package ROI handler initialized")
    
    def extract_field_rois(self, image, template_id, homography):
        """
        Extract ROIs for all fields from a package image.
        
        Args:
            image (numpy.ndarray): Package image
            template_id (str): Template ID
            homography (numpy.ndarray): Homography matrix
            
        Returns:
            dict: Extracted ROIs for each field
        """
        # Apply standard ROI mapping
        rois = self.roi_mapper.apply_mapping(image, template_id, homography, is_sheet=False)
        
        if not rois:
            logger.warning(f"No ROIs defined for package template {template_id}")
            return None
        
        # Apply package-specific enhancements
        enhanced_rois = self._enhance_package_rois(rois)
        
        return enhanced_rois
    
    def _enhance_package_rois(self, rois):
        """
        Apply package-specific enhancements to ROIs.
        
        Args:
            rois (dict): Original extracted ROIs
            
        Returns:
            dict: Enhanced ROIs
        """
        enhanced_rois = {}
        
        for field_name, field_rois in rois.items():
            enhanced_field_rois = []
            
            for roi_data in field_rois:
                roi_image = roi_data['image']
                
                # Apply reflective surface handling if needed
                if self.handle_reflective:
                    roi_image = self._handle_reflective_surface(roi_image)
                
                # Create enhanced ROI data
                enhanced_roi = roi_data.copy()
                enhanced_roi['image'] = roi_image
                
                # Add specific OCR instructions for packages
                if 'ocr_config' not in enhanced_roi or not enhanced_roi['ocr_config']:
                    enhanced_roi['ocr_config'] = self._get_default_package_ocr_config(field_name)
                
                enhanced_field_rois.append(enhanced_roi)
            
            enhanced_rois[field_name] = enhanced_field_rois
        
        return enhanced_rois
    
    def _handle_reflective_surface(self, image):
        """
        Process image to reduce problems with reflective surfaces.
        
        Args:
            image (numpy.ndarray): ROI image
            
        Returns:
            numpy.ndarray: Processed image
        """
        if image is None or image.size == 0:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
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
        
        # Return enhanced image with 3 channels if input had 3 channels
        if len(image.shape) == 3:
            return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            return processed
    
    def _get_default_package_ocr_config(self, field_name):
        """
        Get default OCR configuration for package fields.
        
        Args:
            field_name (str): Field name
            
        Returns:
            dict: OCR configuration
        """
        # Base package OCR configuration
        base_config = {
            'det_db_thresh': 0.3,
            'det_db_box_thresh': 0.4,
            'use_dilation': False,  # Better for isolated text on packages
            'enhance_contrast': True,
            'preprocess_reflection_removal': True
        }
        
        # Field-specific configurations
        if field_name == ROIMapper.COMMERCIAL_NAME:
            return {
                **base_config,
                'det_db_thresh': 0.3,  # More sensitive detection for brand names
                'min_text_confidence': 0.7,
                'detect_orientation': True,  # Brand names might be rotated
                'priority': 3
            }
        elif field_name == ROIMapper.SCIENTIFIC_NAME:
            return {
                **base_config,
                'detect_curved_text': True,  # Scientific names sometimes follow curve
                'min_text_size': 8,  # Scientific names often smaller
                'min_text_confidence': 0.6,
                'priority': 2
            }
        elif field_name == ROIMapper.MANUFACTURER:
            return {
                **base_config,
                'det_db_unclip_ratio': 1.8,  # Manufacturer logos need larger regions
                'min_text_confidence': 0.65,
                'priority': 1
            }
        elif field_name == ROIMapper.DOSAGE:
            return {
                **base_config,
                'detect_digits': True,  # Optimize for number detection
                'det_db_thresh': 0.25,  # More sensitive for dosage numbers
                'min_text_confidence': 0.7,
                'priority': 2
            }
        else:
            return base_config
    
    def handle_missing_tablets(self, image, template_id, homography):
        """
        Process image to account for missing tablets in blister packs.
        
        Args:
            image (numpy.ndarray): Package image
            template_id (str): Template ID
            homography (numpy.ndarray): Homography matrix
            
        Returns:
            dict: Extracted ROIs with missing tablet compensation
        """
        # Extract standard ROIs
        rois = self.extract_field_rois(image, template_id, homography)
        
        if not rois:
            return None
        
        # Detect missing tablets
        missing_areas = self._detect_missing_tablets(image)
        
        if not missing_areas:
            return rois  # No missing tablets detected
        
        # Adjust ROIs to avoid missing tablet areas
        for field_name, field_rois in rois.items():
            for i, roi_data in enumerate(field_rois):
                region = roi_data['region']
                
                # Check if ROI overlaps with missing areas
                overlaps = False
                for missing_area in missing_areas:
                    if self._regions_overlap(region, missing_area):
                        overlaps = True
                        break
                
                if overlaps:
                    # Reduce priority of affected ROI
                    rois[field_name][i]['priority'] -= 1
                    logger.debug(f"Reduced priority of {field_name} ROI due to missing tablet overlap")
        
        return rois
    
    def _detect_missing_tablets(self, image):
        """
        Detect missing tablets in a blister pack.
        
        Args:
            image (numpy.ndarray): Package image
            
        Returns:
            list: List of regions with missing tablets
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        missing_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Missing tablets typically create distinctive circular/oval holes
            # Filter by area and circularity
            if area > 200 and area < 5000:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.5:  # Circular shapes have circularity close to 1
                    x, y, w, h = cv2.boundingRect(contour)
                    missing_areas.append([x, y, x + w, y + h])
        
        return missing_areas
    
    def _regions_overlap(self, region1, region2):
        """
        Check if two regions overlap.
        
        Args:
            region1 (list): First region [x1, y1, x2, y2]
            region2 (list): Second region [x1, y1, x2, y2]
            
        Returns:
            bool: True if regions overlap, False otherwise
        """
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        # Check if one rectangle is to the left of the other
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        
        # Check if one rectangle is above the other
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        
        # Otherwise, rectangles overlap
        return True
