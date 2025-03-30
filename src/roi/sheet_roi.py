"""
Specialized ROI handling for pharmaceutical information sheets.
"""

import cv2
import numpy as np
from loguru import logger

from .roi_mapper import ROIMapper

class SheetROI:
    """
    Handles ROI operations specifically for pharmaceutical information sheets.
    Provides specialized extraction methods for sheet-specific challenges:
    - Dense text formatting
    - Multiple text instances with the same information
    - Hierarchical formatting with sections and subsections
    - Multilingual content including Arabic
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize sheet ROI handler.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.roi_mapper = ROIMapper(config_path)
        
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.special_cases = config['special_cases']
        
        # Sheet-specific extraction settings
        self.handle_multilingual = self.special_cases.get('handle_multilingual', True)
        self.arabic_support = self.special_cases.get('arabic_support', True)
        
        logger.info("Sheet ROI handler initialized")
    
    def extract_field_rois(self, image, template_id, homography):
        """
        Extract ROIs for all fields from an information sheet image.
        
        Args:
            image (numpy.ndarray): Sheet image
            template_id (str): Template ID
            homography (numpy.ndarray): Homography matrix
            
        Returns:
            dict: Extracted ROIs for each field
        """
        # Apply standard ROI mapping
        rois = self.roi_mapper.apply_mapping(image, template_id, homography, is_sheet=True)
        
        if not rois:
            logger.warning(f"No ROIs defined for sheet template {template_id}")
            return None
        
        # Apply sheet-specific enhancements
        enhanced_rois = self._enhance_sheet_rois(rois)
        
        # Extract text sections for contextual understanding
        sections = self._extract_sections(image)
        
        # Enhance ROIs with section context
        for field_name, field_rois in enhanced_rois.items():
            for i, roi_data in enumerate(field_rois):
                region = roi_data['region']
                
                # Find section containing this ROI
                section_context = self._find_containing_section(region, sections)
                
                if section_context:
                    enhanced_rois[field_name][i]['section_context'] = section_context
        
        return enhanced_rois
    
    def _enhance_sheet_rois(self, rois):
        """
        Apply sheet-specific enhancements to ROIs.
        
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
                
                # Apply sheet-specific preprocessing
                roi_image = self._preprocess_sheet_roi(roi_image)
                
                # Create enhanced ROI data
                enhanced_roi = roi_data.copy()
                enhanced_roi['image'] = roi_image
                
                # Add specific OCR instructions for sheets
                if 'ocr_config' not in enhanced_roi or not enhanced_roi['ocr_config']:
                    enhanced_roi['ocr_config'] = self._get_default_sheet_ocr_config(field_name)
                
                enhanced_field_rois.append(enhanced_roi)
            
            enhanced_rois[field_name] = enhanced_field_rois
        
        return enhanced_rois
    
    def _preprocess_sheet_roi(self, image):
        """
        Preprocess ROI image for better text extraction from sheets.
        
        Args:
            image (numpy.ndarray): ROI image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        if image is None or image.size == 0:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to preserve edges while removing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply mild sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(filtered, -1, kernel)
        
        # Return enhanced image with 3 channels if input had 3 channels
        if len(image.shape) == 3:
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        else:
            return sharpened
    
    def _get_default_sheet_ocr_config(self, field_name):
        """
        Get default OCR configuration for sheet fields.
        
        Args:
            field_name (str): Field name
            
        Returns:
            dict: OCR configuration
        """
        # Base sheet OCR configuration
        base_config = {
            'det_db_thresh': 0.2,  # Lower threshold for dense text
            'det_db_box_thresh': 0.55,  # Higher box threshold for cleaner paragraphs
            'use_dilation': True,  # Connect characters in dense text
            'detect_paragraphs': True,
            'group_text_lines': True
        }
        
        # Field-specific configurations
        if field_name == ROIMapper.COMMERCIAL_NAME:
            return {
                **base_config,
                'det_db_thresh': 0.3,  # Commercial name usually more prominent
                'min_text_confidence': 0.8,
                'detect_headers': True,  # Commercial names are often headers
                'priority': 3
            }
        elif field_name == ROIMapper.SCIENTIFIC_NAME:
            return {
                **base_config,
                'detect_text_hierarchy': True,  # Scientific names often in specific sections
                'min_text_confidence': 0.7,
                'priority': 2
            }
        elif field_name == ROIMapper.MANUFACTURER:
            return {
                **base_config,
                'det_db_thresh': 0.25,  # Manufacturer info may be smaller
                'min_text_confidence': 0.7,
                'priority': 1
            }
        elif field_name == ROIMapper.DOSAGE:
            return {
                **base_config,
                'detect_digits': True,  # Optimize for number detection
                'detect_tables': True,  # Dosage info often in tables
                'min_text_confidence': 0.75,
                'priority': 2
            }
        else:
            return base_config
    
    def _extract_sections(self, image):
        """
        Extract text sections from the sheet for contextual understanding.
        
        Args:
            image (numpy.ndarray): Sheet image
            
        Returns:
            list: List of detected sections, each with region and probable content type
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding to detect text regions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to group text into sections
        kernel = np.ones((5, 20), np.uint8)  # Horizontal kernel to connect text lines
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours representing sections
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area to eliminate noise
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Try to categorize the section
                section_type = self._categorize_section(gray[y:y+h, x:x+w])
                
                sections.append({
                    'region': [x, y, x + w, y + h],
                    'type': section_type
                })
        
        return sections
    
    def _categorize_section(self, section_image):
        """
        Categorize a section based on its content.
        
        Args:
            section_image (numpy.ndarray): Section image
            
        Returns:
            str: Section type ('header', 'paragraph', 'table', 'unknown')
        """
        # Simple heuristic-based categorization
        h, w = section_image.shape
        aspect_ratio = w / h if h > 0 else 0
        
        # Headers tend to be wide and short
        if aspect_ratio > 5 and h < 100:
            return 'header'
        
        # Detect tables using Hough lines
        edges = cv2.Canny(section_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 5:
            # Count horizontal and vertical lines
            horizontal_count = 0
            vertical_count = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:
                    horizontal_count += 1
                elif 70 < angle < 110:
                    vertical_count += 1
            
            # Tables typically have both horizontal and vertical lines
            if horizontal_count > 2 and vertical_count > 2:
                return 'table'
        
        # Default to paragraph
        return 'paragraph'
    
    def _find_containing_section(self, region, sections):
        """
        Find the section containing a region.
        
        Args:
            region (list): Region coordinates [x1, y1, x2, y2]
            sections (list): List of detected sections
            
        Returns:
            dict: Section data, or None if not found
        """
        x1, y1, x2, y2 = region
        roi_center_x = (x1 + x2) / 2
        roi_center_y = (y1 + y2) / 2
        
        for section in sections:
            sx1, sy1, sx2, sy2 = section['region']
            
            # Check if ROI center is within section
            if sx1 <= roi_center_x <= sx2 and sy1 <= roi_center_y <= sy2:
                return section
        
        return None
    
    def find_redundant_information(self, rois, ocr_results):
        """
        Find redundant information across different ROIs in a sheet.
        Used for verification and selecting the most reliable instance.
        
        Args:
            rois (dict): Extracted ROIs for each field
            ocr_results (dict): OCR results for each field
            
        Returns:
            dict: Field values with confidence scores, using the most reliable instance
        """
        reliable_values = {}
        
        for field_name, field_results in ocr_results.items():
            if not field_results:
                continue
            
            # Group results by similar values
            value_groups = self._group_similar_values(field_results)
            
            if not value_groups:
                continue
            
            # Find the group with the highest aggregate confidence
            best_group = max(value_groups, key=lambda g: sum(r['confidence'] for r in g) / len(g))
            
            # Calculate average confidence for the group
            avg_confidence = sum(r['confidence'] for r in best_group) / len(best_group)
            
            # Use value from highest confidence result in the group
            best_result = max(best_group, key=lambda r: r['confidence'])
            
            reliable_values[field_name] = {
                'value': best_result['text'],
                'confidence': avg_confidence,
                'instances': len(best_group)
            }
        
        return reliable_values
    
    def _group_similar_values(self, results, similarity_threshold=0.8):
        """
        Group similar text values from OCR results.
        
        Args:
            results (list): List of OCR results
            similarity_threshold (float): Similarity threshold for grouping
            
        Returns:
            list: List of groups, where each group is a list of similar results
        """
        if not results:
            return []
        
        # Sort by confidence (descending)
        sorted_results = sorted(results, key=lambda r: r.get('confidence', 0), reverse=True)
        
        groups = []
        for result in sorted_results:
            text = result.get('text', '').strip().lower()
            if not text:
                continue
            
            # Try to find a matching group
            matched = False
            for group in groups:
                # Compare with first item in the group
                group_text = group[0].get('text', '').strip().lower()
                
                if self._text_similarity(text, group_text) >= similarity_threshold:
                    group.append(result)
                    matched = True
                    break
            
            # If no match found, create a new group
            if not matched:
                groups.append([result])
        
        return groups
    
    def _text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Simple Jaccard similarity on words
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
