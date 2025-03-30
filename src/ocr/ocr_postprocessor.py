"""
OCR postprocessing module for improving text recognition results.
"""

import re
import numpy as np
from loguru import logger

class OCRPostprocessor:
    """
    Provides text postprocessing to improve OCR quality for pharmaceutical content.
    Implements specialized postprocessing for different field types.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize OCR postprocessor.
        
        Args:
            config_path (str): Path to configuration file
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.extraction_config = config['extraction']
        
        # Configure confidence thresholds
        self.commercial_name_threshold = self.extraction_config.get('commercial_name_confidence_threshold', 0.95)
        self.other_fields_threshold = self.extraction_config.get('other_fields_confidence_threshold', 0.90)
        
        logger.info("OCR postprocessor initialized")
    
    def postprocess(self, text, field_type, confidence=None):
        """
        Apply field-specific postprocessing to OCR text.
        
        Args:
            text (str): OCR extracted text
            field_type (str): Field type (commercial_name, scientific_name, manufacturer, dosage)
            confidence (float, optional): OCR confidence for the text
            
        Returns:
            tuple: (processed_text, adjusted_confidence)
        """
        if text is None:
            return None, 0.0
        
        # Trim whitespace
        processed = text.strip()
        
        if not processed:
            return None, 0.0
        
        # Initialize adjusted confidence
        adjusted_confidence = confidence if confidence is not None else 0.5
        
        # Apply field-specific processing
        if field_type == 'commercial_name':
            processed, adjusted_confidence = self._process_commercial_name(processed, adjusted_confidence)
        elif field_type == 'scientific_name':
            processed, adjusted_confidence = self._process_scientific_name(processed, adjusted_confidence)
        elif field_type == 'manufacturer':
            processed, adjusted_confidence = self._process_manufacturer(processed, adjusted_confidence)
        elif field_type == 'dosage':
            processed, adjusted_confidence = self._process_dosage(processed, adjusted_confidence)
        
        return processed, adjusted_confidence
    
    def _process_commercial_name(self, text, confidence):
        """
        Process commercial name text.
        
        Args:
            text (str): Commercial name text
            confidence (float): Initial confidence
            
        Returns:
            tuple: (processed_text, adjusted_confidence)
        """
        # Remove common OCR errors in commercial names
        processed = text
        
        # Convert to uppercase as commercial names are typically capitalized
        processed = processed.upper()
        
        # Remove unwanted characters
        processed = re.sub(r'[^\w\s-]', '', processed)
        
        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Check for common patterns in brand names
        if re.match(r'^[A-Z][A-Z0-9\s-]{2,20}$', processed):
            # Looks like a valid brand name
            adjusted_confidence = min(confidence * 1.1, 1.0)
        else:
            # Doesn't match typical brand name pattern
            adjusted_confidence = confidence * 0.9
        
        return processed, adjusted_confidence
    
    def _process_scientific_name(self, text, confidence):
        """
        Process scientific name (active ingredients) text.
        
        Args:
            text (str): Scientific name text
            confidence (float): Initial confidence
            
        Returns:
            tuple: (processed_text, adjusted_confidence)
        """
        # Scientific names often have specific patterns
        processed = text
        
        # Remove unwanted characters
        processed = re.sub(r'[^\w\s,.()-]', '', processed)
        
        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Check for common active ingredient patterns
        # Most active ingredients end with specific suffixes
        common_suffixes = [
            'acetate', 'sodium', 'hydrochloride', 'hcl', 'citrate', 'sulfate', 
            'phosphate', 'tartrate', 'maleate', 'nitrate', 'oxide', 'chloride'
        ]
        
        # Check if text contains common scientific term patterns
        has_scientific_pattern = False
        
        # Check for suffix
        for suffix in common_suffixes:
            if suffix in processed.lower():
                has_scientific_pattern = True
                break
        
        # Check for dosage pattern in the text (e.g., "500 mg")
        if re.search(r'\d+\s*(?:mg|g|mcg|µg|ml)', processed.lower()):
            has_scientific_pattern = True
        
        if has_scientific_pattern:
            adjusted_confidence = min(confidence * 1.1, 1.0)
        else:
            adjusted_confidence = confidence * 0.9
        
        return processed, adjusted_confidence
    
    def _process_manufacturer(self, text, confidence):
        """
        Process manufacturer name text.
        
        Args:
            text (str): Manufacturer name text
            confidence (float): Initial confidence
            
        Returns:
            tuple: (processed_text, adjusted_confidence)
        """
        processed = text
        
        # Remove unwanted characters
        processed = re.sub(r'[^\w\s,.-]', '', processed)
        
        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Common terms indicating manufacturer
        common_terms = ['inc', 'corp', 'llc', 'ltd', 'pharmaceutical', 'pharma', 'laboratories', 'labs', 'company']
        
        # Check if text contains common manufacturer term patterns
        has_company_pattern = False
        for term in common_terms:
            if term in processed.lower():
                has_company_pattern = True
                break
        
        if has_company_pattern:
            adjusted_confidence = min(confidence * 1.05, 1.0)
        else:
            adjusted_confidence = confidence
        
        return processed, adjusted_confidence
    
    def _process_dosage(self, text, confidence):
        """
        Process dosage text.
        
        Args:
            text (str): Dosage text
            confidence (float): Initial confidence
            
        Returns:
            tuple: (processed_text, adjusted_confidence)
        """
        processed = text
        
        # Extract dosage value and unit
        # Look for patterns like "500 mg", "10 mL", etc.
        dosage_match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', processed)
        
        if dosage_match:
            value, unit = dosage_match.groups()
            
            # Normalize units
            unit_lower = unit.lower()
            if unit_lower in ['mg', 'mgs', 'milligram', 'milligrams']:
                unit = 'mg'
            elif unit_lower in ['g', 'gm', 'gram', 'grams']:
                unit = 'g'
            elif unit_lower in ['mcg', 'µg', 'microgram', 'micrograms']:
                unit = 'mcg'
            elif unit_lower in ['ml', 'milliliter', 'milliliters', 'millilitre']:
                unit = 'mL'
            
            # Format nicely
            processed = f"{value} {unit}"
            adjusted_confidence = min(confidence * 1.2, 1.0)  # Boost confidence
        else:
            # Doesn't look like a dosage
            adjusted_confidence = confidence * 0.8
        
        return processed, adjusted_confidence
    
    def postprocess_results(self, results, field_type):
        """
        Apply postprocessing to OCR results for a specific field.
        
        Args:
            results (dict): OCR results with values and confidences
            field_type (str): Field type
            
        Returns:
            dict: Postprocessed results
        """
        # Process single result
        if isinstance(results, dict) and 'value' in results:
            value = results['value']
            confidence = results.get('confidence', 0.0)
            
            processed_value, adjusted_confidence = self.postprocess(value, field_type, confidence)
            
            return {
                'value': processed_value,
                'confidence': adjusted_confidence,
                'source_roi': results.get('source_roi')
            }
        
        # Process list of results
        if isinstance(results, list):
            processed_results = []
            
            for result in results:
                if isinstance(result, dict) and 'text' in result:
                    text = result['text']
                    confidence = result.get('confidence', 0.0)
                    
                    processed_text, adjusted_confidence = self.postprocess(text, field_type, confidence)
                    
                    processed_result = result.copy()
                    processed_result['text'] = processed_text
                    processed_result['confidence'] = adjusted_confidence
                    
                    processed_results.append(processed_result)
                else:
                    processed_results.append(result)
            
            return processed_results
        
        return results
    
    def filter_by_confidence(self, results, field_type):
        """
        Filter results by confidence threshold based on field type.
        
        Args:
            results (list): OCR results
            field_type (str): Field type
            
        Returns:
            list: Filtered results
        """
        if not isinstance(results, list):
            return results
        
        # Determine threshold
        if field_type == 'commercial_name':
            threshold = self.commercial_name_threshold
        else:
            threshold = self.other_fields_threshold
        
        # Filter results
        filtered_results = [r for r in results if r.get('confidence', 0) >= threshold]
        
        return filtered_results if filtered_results else results  # Return original if all filtered out
    
    def merge_text_lines(self, results):
        """
        Merge multiple text lines into a coherent paragraph.
        Used for information sheets with multi-line content.
        
        Args:
            results (list): OCR results with multiple text lines
            
        Returns:
            dict: Merged result
        """
        if not results:
            return {'value': None, 'confidence': 0.0}
        
        # Sort by vertical position (assuming box coordinates are available)
        sorted_results = []
        
        for r in results:
            if 'box' in r:
                # Calculate center y-coordinate
                box = np.array(r['box'])
                center_y = np.mean(box[:, 1])
                
                sorted_results.append({
                    'text': r.get('text', ''),
                    'confidence': r.get('confidence', 0.0),
                    'center_y': center_y
                })
            else:
                sorted_results.append({
                    'text': r.get('text', ''),
                    'confidence': r.get('confidence', 0.0),
                    'center_y': 0  # Default position
                })
        
        # Sort by vertical position
        sorted_results.sort(key=lambda x: x['center_y'])
        
        # Merge text
        merged_text = ' '.join(r['text'] for r in sorted_results if r['text'])
        
        # Calculate average confidence
        confidences = [r['confidence'] for r in sorted_results if r['confidence'] > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'value': merged_text,
            'confidence': avg_confidence
        }
    
    def extract_numeric_value(self, text):
        """
        Extract numeric value from text.
        
        Args:
            text (str): Input text
            
        Returns:
            float or None: Extracted numeric value or None if not found
        """
        if not text:
            return None
        
        # Try to find a numeric value
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        
        return None
    
    def extract_units(self, text):
        """
        Extract units from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str or None: Extracted units or None if not found
        """
        if not text:
            return None
        
        # Common pharmaceutical units
        units = ['mg', 'g', 'mcg', 'µg', 'ml', 'mL', '%', 'IU']
        
        # Try to find units
        for unit in units:
            if unit in text:
                return unit
        
        # Try using regex for more complex cases
        match = re.search(r'\d+\s*([a-zA-Z%]+)', text)
        
        if match:
            return match.group(1)
        
        return None
