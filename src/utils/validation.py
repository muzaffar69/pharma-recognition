"""
Validation utilities for pharmaceutical package and sheet recognition.
"""

import re
from loguru import logger
import numpy as np

class ValidationUtils:
    """
    Validation utilities for verifying extracted information from pharmaceutical products.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize validation utilities.
        
        Args:
            config_path (str): Path to configuration file
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.extraction_config = config['extraction']
        
        # Configure thresholds
        self.commercial_name_threshold = self.extraction_config.get('commercial_name_confidence_threshold', 0.95)
        self.other_fields_threshold = self.extraction_config.get('other_fields_confidence_threshold', 0.90)
        self.use_verification = self.extraction_config.get('use_verification', True)
        
        logger.info("Validation utilities initialized")
    
    def validate_commercial_name(self, name, confidence):
        """
        Validate commercial name.
        
        Args:
            name (str): Commercial name
            confidence (float): OCR confidence
            
        Returns:
            tuple: (is_valid, confidence_adjustment)
        """
        if name is None or not name.strip():
            return False, 0.0
        
        # Confidence check
        if confidence < self.commercial_name_threshold:
            logger.debug(f"Commercial name confidence too low: {confidence:.3f} < {self.commercial_name_threshold:.3f}")
            return False, 0.0
        
        # Format validation
        if not self._is_valid_commercial_name_format(name):
            logger.debug(f"Invalid commercial name format: {name}")
            return False, -0.1
        
        # Length validation
        if len(name) < 2 or len(name) > 50:
            logger.debug(f"Commercial name length invalid: {len(name)} characters")
            return False, -0.1
        
        # All validations passed
        return True, 0.0
    
    def validate_scientific_name(self, name, confidence):
        """
        Validate scientific name (active ingredients).
        
        Args:
            name (str): Scientific name
            confidence (float): OCR confidence
            
        Returns:
            tuple: (is_valid, confidence_adjustment)
        """
        if name is None or not name.strip():
            return False, 0.0
        
        # Confidence check
        if confidence < self.other_fields_threshold:
            logger.debug(f"Scientific name confidence too low: {confidence:.3f} < {self.other_fields_threshold:.3f}")
            return False, 0.0
        
        # Format validation
        if not self._is_valid_scientific_name_format(name):
            logger.debug(f"Invalid scientific name format: {name}")
            return False, -0.1
        
        # Length validation
        if len(name) < 3 or len(name) > 100:
            logger.debug(f"Scientific name length invalid: {len(name)} characters")
            return False, -0.1
        
        # All validations passed
        return True, 0.0
    
    def validate_manufacturer(self, name, confidence):
        """
        Validate manufacturer name.
        
        Args:
            name (str): Manufacturer name
            confidence (float): OCR confidence
            
        Returns:
            tuple: (is_valid, confidence_adjustment)
        """
        if name is None or not name.strip():
            return False, 0.0
        
        # Confidence check
        if confidence < self.other_fields_threshold:
            logger.debug(f"Manufacturer confidence too low: {confidence:.3f} < {self.other_fields_threshold:.3f}")
            return False, 0.0
        
        # Format validation
        if not self._is_valid_manufacturer_format(name):
            logger.debug(f"Invalid manufacturer format: {name}")
            return False, -0.1
        
        # Length validation
        if len(name) < 2 or len(name) > 100:
            logger.debug(f"Manufacturer length invalid: {len(name)} characters")
            return False, -0.1
        
        # All validations passed
        return True, 0.0
    
    def validate_dosage(self, dosage, confidence):
        """
        Validate dosage information.
        
        Args:
            dosage (str): Dosage information
            confidence (float): OCR confidence
            
        Returns:
            tuple: (is_valid, confidence_adjustment)
        """
        if dosage is None or not dosage.strip():
            return False, 0.0
        
        # Confidence check
        if confidence < self.other_fields_threshold:
            logger.debug(f"Dosage confidence too low: {confidence:.3f} < {self.other_fields_threshold:.3f}")
            return False, 0.0
        
        # Format validation
        if not self._is_valid_dosage_format(dosage):
            logger.debug(f"Invalid dosage format: {dosage}")
            return False, -0.1
        
        # All validations passed
        return True, 0.0
    
    def _is_valid_commercial_name_format(self, name):
        """
        Check if commercial name has valid format.
        
        Args:
            name (str): Commercial name
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Remove common OCR errors and normalize
        name = name.strip()
        
        # Common format for commercial names
        # Usually consists of letters, numbers, and some special characters
        # Often starts with a capital letter
        # Examples: Advil, Tylenol, Zithromax, Lipitor
        pattern = r'^[A-Za-z0-9][A-Za-z0-9\s\-\'®™]+$'
        
        return bool(re.match(pattern, name))
    
    def _is_valid_scientific_name_format(self, name):
        """
        Check if scientific name has valid format.
        
        Args:
            name (str): Scientific name
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Remove common OCR errors and normalize
        name = name.strip()
        
        # Scientific names often include chemical names, latin terms, etc.
        # More permissive pattern
        pattern = r'^[A-Za-z0-9][A-Za-z0-9\s\-\(\),.]+$'
        
        return bool(re.match(pattern, name))
    
    def _is_valid_manufacturer_format(self, name):
        """
        Check if manufacturer name has valid format.
        
        Args:
            name (str): Manufacturer name
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Remove common OCR errors and normalize
        name = name.strip()
        
        # Manufacturer names often include company suffixes
        # Examples: Pfizer Inc., Johnson & Johnson, Bayer AG
        pattern = r'^[A-Za-z0-9][A-Za-z0-9\s\-\'\.,&]+$'
        
        return bool(re.match(pattern, name))
    
    def _is_valid_dosage_format(self, dosage):
        """
        Check if dosage has valid format.
        
        Args:
            dosage (str): Dosage information
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Remove common OCR errors and normalize
        dosage = dosage.strip()
        
        # Common dosage formats:
        # Examples: 500 mg, 10 mL, 250 mcg
        pattern = r'^\d+(\.\d+)?\s*(mg|g|mcg|µg|ml|mL|%|IU)$'
        
        return bool(re.match(pattern, dosage))
    
    def verify_with_database(self, results, db_connector):
        """
        Verify extracted information against database records.
        
        Args:
            results (dict): Extraction results
            db_connector: Database connector
            
        Returns:
            dict: Verified results with confidence adjustments
        """
        if not self.use_verification or db_connector is None:
            return results
        
        verified_results = {}
        
        # Get commercial name
        commercial_name = results.get('commercial_name', {}).get('value')
        
        if commercial_name:
            # Look up product in database
            product = db_connector.get_product_by_name(commercial_name)
            
            if product:
                logger.debug(f"Found matching product in database: {commercial_name}")
                
                # Verify each field against database record
                for field in ['commercial_name', 'scientific_name', 'manufacturer', 'dosage']:
                    if field in results:
                        field_result = results[field].copy()
                        extracted_value = field_result.get('value')
                        db_value = product.get(field)
                        
                        if extracted_value and db_value:
                            similarity = self._calculate_text_similarity(extracted_value, db_value)
                            
                            # Update confidence based on similarity
                            if similarity > 0.8:
                                # Boost confidence for high similarity
                                field_result['confidence'] = min(field_result.get('confidence', 0) * 1.2, 1.0)
                                field_result['verified'] = True
                            elif similarity < 0.3:
                                # Reduce confidence for low similarity
                                field_result['confidence'] = field_result.get('confidence', 0) * 0.8
                                field_result['verified'] = False
                            
                            field_result['similarity'] = similarity
                        
                        verified_results[field] = field_result
                    else:
                        verified_results[field] = {'value': product.get(field), 'confidence': 0.99, 'source': 'database'}
                
                # Add template ID from database
                verified_results['template_id'] = product.get('template_id')
                
                return verified_results
        
        # No database match found, return original results
        return results
    
    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Simple Jaccard similarity for words
        words1 = set(re.findall(r'\w+', text1))
        words2 = set(re.findall(r'\w+', text2))
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def verify_extraction_results(self, results):
        """
        Verify extraction results against validation rules.
        
        Args:
            results (dict): Extraction results
            
        Returns:
            dict: Validated results with confidence adjustments
        """
        validated_results = {}
        
        for field_name, field_data in results.items():
            if not isinstance(field_data, dict):
                validated_results[field_name] = field_data
                continue
            
            value = field_data.get('value')
            confidence = field_data.get('confidence', 0.0)
            
            # Skip non-extracted fields
            if value is None:
                validated_results[field_name] = field_data
                continue
            
            # Validate based on field type
            is_valid = True
            confidence_adjustment = 0.0
            
            if field_name == 'commercial_name':
                is_valid, confidence_adjustment = self.validate_commercial_name(value, confidence)
            elif field_name == 'scientific_name':
                is_valid, confidence_adjustment = self.validate_scientific_name(value, confidence)
            elif field_name == 'manufacturer':
                is_valid, confidence_adjustment = self.validate_manufacturer(value, confidence)
            elif field_name == 'dosage':
                is_valid, confidence_adjustment = self.validate_dosage(value, confidence)
            
            # Apply validation results
            validated_field = field_data.copy()
            validated_field['valid'] = is_valid
            validated_field['confidence'] = min(1.0, max(0.0, confidence + confidence_adjustment))
            
            validated_results[field_name] = validated_field
        
        return validated_results
    
    def calculate_overall_confidence(self, results):
        """
        Calculate overall confidence score for extraction results.
        
        Args:
            results (dict): Extraction results
            
        Returns:
            float: Overall confidence score (0-1)
        """
        # Extract confidence scores for each field
        confidence_scores = []
        
        for field_name, field_data in results.items():
            if isinstance(field_data, dict) and 'confidence' in field_data:
                # Use confidence and field importance to weight the score
                if field_name == 'commercial_name':
                    weight = 0.4  # Highest importance
                elif field_name in ['scientific_name', 'dosage']:
                    weight = 0.25  # Medium importance
                else:
                    weight = 0.1  # Lower importance
                
                confidence_scores.append((field_data['confidence'], weight))
        
        # Calculate weighted average
        if not confidence_scores:
            return 0.0
        
        total_weight = sum(weight for _, weight in confidence_scores)
        weighted_sum = sum(conf * weight for conf, weight in confidence_scores)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
