"""
Specialized OCR processor for pharmaceutical information sheets.
"""

import cv2
import numpy as np
from loguru import logger
import re

from .ocr_engine import OCREngine
from .ocr_preprocessor import OCRPreprocessor
from .ocr_postprocessor import OCRPostprocessor
from ..utils.performance_monitor import PerformanceTimer

class SheetProcessor:
    """
    Specialized OCR processor for pharmaceutical information sheets.
    Handles sheet-specific challenges such as:
    - Dense text layout
    - Multiple instances of the same information
    - Hierarchical formatting with sections
    - Tables and bullets
    - Multilingual content including Arabic
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize sheet processor.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        
        # Initialize components
        self.ocr_engine = OCREngine(config_path)
        self.preprocessor = OCRPreprocessor(config_path)
        self.postprocessor = OCRPostprocessor(config_path)
        
        # Configuration
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.special_cases = config['special_cases']
        
        # Special case handling
        self.handle_multilingual = self.special_cases.get('handle_multilingual', True)
        self.arabic_support = self.special_cases.get('arabic_support', True)
        
        # Performance timer
        self.timer = PerformanceTimer()
        
        logger.info("Sheet processor initialized")
    
    def process(self, image, rois):
        """
        Process information sheet image with ROIs to extract information.
        
        Args:
            image (numpy.ndarray): Sheet image
            rois (dict): ROIs for each field
            
        Returns:
            dict: Extracted field data
        """
        if image is None or rois is None:
            logger.warning("Invalid inputs for sheet processing")
            return {}
        
        self.timer.start("sheet_processing")
        
        # Detect language to optimize processing
        languages = self.ocr_engine.detect_language(image)
        has_arabic = 'ar' in languages and languages['ar'] > 0.3
        
        # Process each field
        results = {}
        
        for field_name, field_rois in rois.items():
            field_results = self._process_field(image, field_rois, field_name, has_arabic)
            results[field_name] = field_results
        
        # For sheets, check for redundant information for verification
        if len(results) > 0:
            sheet_roi_handler = __import__('src.roi.sheet_roi', fromlist=['SheetROI']).SheetROI(self.config_path)
            reliable_values = sheet_roi_handler.find_redundant_information(rois, results)
            
            # Update results with verified values from redundant sources
            for field_name, verified_data in reliable_values.items():
                if verified_data['instances'] > 1 and verified_data['confidence'] > results[field_name].get('confidence', 0):
                    results[field_name] = {
                        'value': verified_data['value'],
                        'confidence': verified_data['confidence'],
                        'verified': True,
                        'instances': verified_data['instances']
                    }
        
        elapsed_ms = self.timer.stop("sheet_processing")
        logger.debug(f"Sheet processing completed in {elapsed_ms:.2f}ms")
        
        return results
    
    def _process_field(self, image, field_rois, field_name, has_arabic=False):
        """
        Process a specific field from ROIs.
        
        Args:
            image (numpy.ndarray): Sheet image
            field_rois (list): ROIs for the field
            field_name (str): Field name
            has_arabic (bool): Whether Arabic text was detected
            
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
            section_context = roi_data.get('section_context', None)
            
            # Skip empty ROI images
            if roi_image is None or roi_image.size == 0:
                continue
            
            # Preprocess based on language
            if has_arabic and self.arabic_support:
                preprocessed = self.preprocessor.preprocess_arabic_text(roi_image)
            else:
                preprocessed = self.preprocessor.preprocess_sheet(roi_image)
            
            # Apply specialized preprocessing based on field type and section context
            preprocessed = self._apply_specialized_preprocessing(preprocessed, field_name, section_context)
            
            # Apply OCR
            try:
                text_results = self.ocr_engine.recognize_text(preprocessed, is_package=False, config_override=ocr_config)
                
                # Add roi_index to each result
                for result in text_results:
                    result['roi_index'] = roi_index
                    result['priority'] = roi_data.get('priority', 0)
                    
                    # Add section context if available
                    if section_context:
                        result['section_type'] = section_context.get('type', 'unknown')
                
                # For ROIs with multiple text lines, merge them if in the same paragraph
                if len(text_results) > 1 and section_context and section_context.get('type') == 'paragraph':
                    merged_result = self.postprocessor.merge_text_lines(text_results)
                    merged_result['roi_index'] = roi_index
                    merged_result['priority'] = roi_data.get('priority', 0)
                    
                    # For merged results, create a single result with the merged text
                    all_results.append({
                        'text': merged_result['value'],
                        'confidence': merged_result['confidence'],
                        'roi_index': roi_index,
                        'priority': roi_data.get('priority', 0),
                        'merged': True
                    })
                else:
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
    
    def _apply_specialized_preprocessing(self, image, field_name, section_context=None):
        """
        Apply specialized preprocessing based on field type and section context.
        
        Args:
            image (numpy.ndarray): ROI image
            field_name (str): Field name
            section_context (dict, optional): Section context information
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Apply field-specific preprocessing
        if field_name == 'commercial_name':
            # For commercial names, check if this is a header section
            if section_context and section_context.get('type') == 'header':
                return self._enhance_header_text(image)
            else:
                return self._enhance_bold_text(image)
                
        elif field_name == 'scientific_name':
            return self._enhance_technical_text(image)
            
        elif field_name == 'manufacturer':
            return image  # Use standard preprocessing
            
        elif field_name == 'dosage':
            # For dosage, check if this is a table section
            if section_context and section_context.get('type') == 'table':
                return self._enhance_table_text(image)
            else:
                return self._enhance_numeric_text(image)
        
        # Default preprocessing
        return image
    
    def _enhance_header_text(self, image):
        """
        Enhance text in header sections.
        
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
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Return in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return binary
    
    def _enhance_bold_text(self, image):
        """
        Enhance bold text (often used for commercial names).
        
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
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding with larger block size for bold text
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
        
        # Return in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return binary
    
    def _enhance_technical_text(self, image):
        """
        Enhance technical text (scientific names, chemical formulas).
        
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
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Return in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            return enhanced
    
    def _enhance_table_text(self, image):
        """
        Enhance text in table sections.
        
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
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Return in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return binary
    
    def _enhance_numeric_text(self, image):
        """
        Enhance numeric text (dosages, measurements).
        
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
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Return in original format
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return binary
    
    def process_sections(self, image, rois):
        """
        Process document by sections to improve contextual understanding.
        
        Args:
            image (numpy.ndarray): Sheet image
            rois (dict): ROIs for each field
            
        Returns:
            dict: Extracted field data with section context
        """
        # Start with standard processing
        results = self.process(image, rois)
        
        # Extract sections from the document
        sheet_roi_handler = __import__('src.roi.sheet_roi', fromlist=['SheetROI']).SheetROI(self.config_path)
        sections = sheet_roi_handler._extract_sections(image)
        
        # For low confidence results, try to find information based on section context
        for field_name, field_result in results.items():
            if field_result.get('confidence', 0) < 0.7:
                # Look for section containing the field information
                target_section_type = self._get_target_section_type(field_name)
                
                if target_section_type:
                    # Find sections matching the target type
                    matching_sections = [s for s in sections if s.get('type') == target_section_type]
                    
                    if matching_sections:
                        # Extract text from these sections
                        section_results = []
                        
                        for section in matching_sections:
                            region = section['region']
                            x1, y1, x2, y2 = region
                            
                            section_image = image[int(y1):int(y2), int(x1):int(x2)]
                            
                            # Preprocess and OCR
                            preprocessed = self.preprocessor.preprocess_sheet(section_image)
                            section_text_results = self.ocr_engine.recognize_text(preprocessed, is_package=False)
                            
                            # Apply field-specific post-processing
                            processed_results = self.postprocessor.postprocess_results(section_text_results, field_name)
                            
                            # Extract relevant information using pattern matching
                            extracted_text = self._extract_field_from_section_text(
                                processed_results, field_name
                            )
                            
                            if extracted_text:
                                section_results.append({
                                    'text': extracted_text,
                                    'confidence': 0.8,  # Default confidence for section extraction
                                    'section_type': target_section_type
                                })
                        
                        # If section-based extraction found results, use the best one
                        if section_results:
                            # Use the first result (could implement more sophisticated selection)
                            results[field_name] = {
                                'value': section_results[0]['text'],
                                'confidence': section_results[0]['confidence'],
                                'source': 'section',
                                'section_type': section_results[0]['section_type']
                            }
        
        return results
    
    def _get_target_section_type(self, field_name):
        """
        Get the most likely section type for a given field.
        
        Args:
            field_name (str): Field name
            
        Returns:
            str: Target section type, or None
        """
        if field_name == 'commercial_name':
            return 'header'
        elif field_name == 'scientific_name':
            return 'paragraph'
        elif field_name == 'manufacturer':
            return 'header'
        elif field_name == 'dosage':
            return 'table'
        
        return None
    
    def _extract_field_from_section_text(self, text_results, field_name):
        """
        Extract specific field information from section text using patterns.
        
        Args:
            text_results (list): OCR results from a section
            field_name (str): Field name
            
        Returns:
            str: Extracted field text, or None
        """
        # Combine all text
        if not text_results:
            return None
        
        combined_text = ' '.join([r.get('text', '') for r in text_results])
        
        # Apply field-specific patterns
        if field_name == 'commercial_name':
            # Look for text in all caps or with trademark symbols
            match = re.search(r'([A-Z]{3,}(?:\s[A-Z]{3,})*|\w+(?:\s\w+){0,2}(?:®|™))', combined_text)
            return match.group(1) if match else None
            
        elif field_name == 'scientific_name':
            # Look for patterns like "contains [ingredient]" or "[ingredient] [dosage]"
            patterns = [
                r'(?:contains|active ingredient(?:s)?:?)\s+([A-Za-z]+(?:\s[A-Za-z]+){0,3})',
                r'([A-Za-z]+(?:\s[A-Za-z]+){0,3})\s+\d+\s*(?:mg|g|mcg|ml)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return None
            
        elif field_name == 'manufacturer':
            # Look for patterns like "manufactured by" or company suffixes
            patterns = [
                r'(?:manufactured|produced|distributed)\s+by\s+([A-Za-z]+(?:\s[A-Za-z]+){0,3})',
                r'([A-Za-z]+(?:\s[A-Za-z]+){0,3})\s+(?:inc|corp|llc|ltd|pharmaceutical|pharma)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return None
            
        elif field_name == 'dosage':
            # Look for dosage patterns
            match = re.search(r'(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml))', combined_text, re.IGNORECASE)
            return match.group(1) if match else None
        
        return None
