"""
Unit tests for OCR module.
"""

import os
import sys
import unittest
import cv2
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components to test
from src.ocr.ocr_engine import OCREngine
from src.ocr.ocr_preprocessor import OCRPreprocessor
from src.ocr.ocr_postprocessor import OCRPostprocessor
from src.ocr.package_processor import PackageProcessor
from src.ocr.sheet_processor import SheetProcessor
from src.roi.roi_mapper import ROIMapper

class TestOCREngine(unittest.TestCase):
    """Test cases for OCREngine class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize OCR engine
        cls.ocr_engine = OCREngine(cls.config_path)
        
        # Create sample test images
        cls.text_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(cls.text_image, "TestDrug 500mg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    def test_initialization(self):
        """Test OCR engine initialization."""
        self.assertIsNotNone(self.ocr_engine)
        self.assertIsNotNone(self.ocr_engine.standard_ocr)
        self.assertIsNotNone(self.ocr_engine.package_ocr)
        self.assertIsNotNone(self.ocr_engine.sheet_ocr)
    
    def test_recognize_text_package(self):
        """Test text recognition for package."""
        results = self.ocr_engine.recognize_text(self.text_image, is_package=True)
        
        # Results should have some text
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for result in results:
            self.assertIn('box', result)
            self.assertIn('text', result)
            self.assertIn('confidence', result)
    
    def test_recognize_text_sheet(self):
        """Test text recognition for sheet."""
        results = self.ocr_engine.recognize_text(self.text_image, is_package=False)
        
        # Results should have some text
        self.assertGreater(len(results), 0)
    
    def test_recognize_text_fast_mode(self):
        """Test text recognition in fast mode."""
        results = self.ocr_engine.recognize_text(self.text_image, is_package=True, fast_mode=True)
        
        # Results should have some text
        self.assertGreater(len(results), 0)
    
    def test_recognize_text_with_config_override(self):
        """Test text recognition with config override."""
        # Custom OCR config
        config_override = {
            'det_db_thresh': 0.4,
            'min_text_confidence': 0.7
        }
        
        results = self.ocr_engine.recognize_text(self.text_image, is_package=True, config_override=config_override)
        
        # Results should have some text
        self.assertGreater(len(results), 0)
    
    def test_extract_field(self):
        """Test extracting field from ROIs."""
        # Create test ROIs
        rois = {
            ROIMapper.COMMERCIAL_NAME: [
                {
                    'image': self.text_image,
                    'region': np.array([0, 0, 300, 100]),
                    'priority': 3
                }
            ]
        }
        
        field_result = self.ocr_engine.extract_field(rois, ROIMapper.COMMERCIAL_NAME, is_package=True)
        
        # Result should have value and confidence
        self.assertIn('value', field_result)
        self.assertIn('confidence', field_result)
    
    def test_extract_fields(self):
        """Test extracting all fields from ROIs."""
        # Create test ROIs
        rois = {
            ROIMapper.COMMERCIAL_NAME: [
                {
                    'image': self.text_image,
                    'region': np.array([0, 0, 300, 100]),
                    'priority': 3
                }
            ],
            ROIMapper.DOSAGE: [
                {
                    'image': self.text_image,
                    'region': np.array([0, 0, 300, 100]),
                    'priority': 2
                }
            ]
        }
        
        results = self.ocr_engine.extract_fields(rois, is_package=True)
        
        # Results should include both fields
        self.assertIn(ROIMapper.COMMERCIAL_NAME, results)
        self.assertIn(ROIMapper.DOSAGE, results)
    
    def test_detect_language(self):
        """Test language detection."""
        languages = self.ocr_engine.detect_language(self.text_image)
        
        # Should detect English
        self.assertIn('en', languages)
        self.assertGreaterEqual(languages['en'], 0.5)
    
    def test_visualize_results(self):
        """Test visualizing OCR results."""
        # Get OCR results
        results = self.ocr_engine.recognize_text(self.text_image, is_package=True)
        
        # Visualize results
        visualization = self.ocr_engine.visualize_results(self.text_image, results)
        
        # Should return an image with same dimensions
        self.assertEqual(visualization.shape, self.text_image.shape)


class TestOCRPreprocessor(unittest.TestCase):
    """Test cases for OCRPreprocessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize OCR preprocessor
        cls.preprocessor = OCRPreprocessor(cls.config_path)
        
        # Create sample test images
        cls.regular_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(cls.regular_image, "TestDrug 500mg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Create reflective surface image
        cls.reflective_image = np.ones((100, 300, 3), dtype=np.uint8) * 200
        # Add gradient to simulate reflection
        for i in range(100):
            cls.reflective_image[i, :, :] = np.minimum(255, 200 + i // 2)
        cv2.putText(cls.reflective_image, "TestDrug 500mg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor)
    
    def test_preprocess_package(self):
        """Test package-specific preprocessing."""
        processed = self.preprocessor.preprocess_package(self.regular_image)
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape, self.regular_image.shape)
    
    def test_preprocess_sheet(self):
        """Test sheet-specific preprocessing."""
        processed = self.preprocessor.preprocess_sheet(self.regular_image)
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape, self.regular_image.shape)
    
    def test_preprocess_with_options(self):
        """Test preprocessing with custom options."""
        options = {
            'enhance_contrast': True,
            'remove_noise': True,
            'handle_reflective': True,
            'handle_perspective': False,
            'sharpen': True,
            'denoise': True
        }
        
        processed = self.preprocessor.preprocess(self.regular_image, is_package=True, preprocessing_options=options)
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape, self.regular_image.shape)
    
    def test_preprocess_arabic_text(self):
        """Test preprocessing for Arabic text."""
        processed = self.preprocessor.preprocess_arabic_text(self.regular_image)
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape, self.regular_image.shape)
    
    def test_handle_reflective_surface(self):
        """Test handling of reflective surfaces."""
        processed = self.preprocessor._handle_reflective_surface(self.reflective_image)
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape[:2], self.reflective_image.shape[:2])
        
        # Should have different pixel values due to processing
        self.assertFalse(np.array_equal(self.reflective_image, processed))
    
    def test_correct_perspective(self):
        """Test perspective correction."""
        # Create image with perspective distortion
        src_points = np.array([
            [50, 50],
            [250, 60],
            [240, 90],
            [40, 80]
        ], dtype=np.float32)
        
        dst_points = np.array([
            [50, 50],
            [250, 50],
            [250, 90],
            [50, 90]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        distorted = cv2.warpPerspective(self.regular_image, np.linalg.inv(M), (300, 100))
        
        processed = self.preprocessor._correct_perspective(distorted)
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape[:2], distorted.shape[:2])


class TestOCRPostprocessor(unittest.TestCase):
    """Test cases for OCRPostprocessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize OCR postprocessor
        cls.postprocessor = OCRPostprocessor(cls.config_path)
    
    def test_initialization(self):
        """Test postprocessor initialization."""
        self.assertIsNotNone(self.postprocessor)
    
    def test_postprocess_commercial_name(self):
        """Test postprocessing commercial name."""
        text = "TESTDRUG"
        confidence = 0.8
        
        processed_text, adjusted_confidence = self.postprocessor.postprocess(
            text, 'commercial_name', confidence
        )
        
        # Text should be uppercase for commercial names
        self.assertEqual(processed_text, text)
        
        # Confidence should be adjusted
        self.assertIsNotNone(adjusted_confidence)
    
    def test_postprocess_scientific_name(self):
        """Test postprocessing scientific name."""
        text = "Acetaminophen"
        confidence = 0.8
        
        processed_text, adjusted_confidence = self.postprocessor.postprocess(
            text, 'scientific_name', confidence
        )
        
        # Text should remain the same
        self.assertEqual(processed_text, text)
        
        # Confidence should be adjusted
        self.assertIsNotNone(adjusted_confidence)
    
    def test_postprocess_manufacturer(self):
        """Test postprocessing manufacturer name."""
        text = "Pharma Inc."
        confidence = 0.8
        
        processed_text, adjusted_confidence = self.postprocessor.postprocess(
            text, 'manufacturer', confidence
        )
        
        # Text should remain the same
        self.assertEqual(processed_text, text)
        
        # Confidence should be adjusted
        self.assertIsNotNone(adjusted_confidence)
    
    def test_postprocess_dosage(self):
        """Test postprocessing dosage."""
        text = "500 mg"
        confidence = 0.8
        
        processed_text, adjusted_confidence = self.postprocessor.postprocess(
            text, 'dosage', confidence
        )
        
        # Text should remain the same
        self.assertEqual(processed_text, text)
        
        # Confidence should be adjusted
        self.assertIsNotNone(adjusted_confidence)
        
        # Test formatting
        text = "500mg"
        processed_text, _ = self.postprocessor.postprocess(text, 'dosage', confidence)
        self.assertEqual(processed_text, "500 mg")
    
    def test_postprocess_results(self):
        """Test postprocessing OCR results."""
        results = [
            {
                'text': 'TESTDRUG',
                'confidence': 0.8,
                'box': np.array([[10, 10], [100, 10], [100, 50], [10, 50]])
            },
            {
                'text': '500 mg',
                'confidence': 0.75,
                'box': np.array([[10, 60], [100, 60], [100, 100], [10, 100]])
            }
        ]
        
        processed_results = self.postprocessor.postprocess_results(results, 'commercial_name')
        
        # Should return list with same length
        self.assertEqual(len(processed_results), len(results))
        
        # Each result should have text and confidence
        for result in processed_results:
            self.assertIn('text', result)
            self.assertIn('confidence', result)
    
    def test_filter_by_confidence(self):
        """Test filtering results by confidence."""
        results = [
            {
                'text': 'TESTDRUG',
                'confidence': 0.95,
                'box': np.array([[10, 10], [100, 10], [100, 50], [10, 50]])
            },
            {
                'text': 'TEST',
                'confidence': 0.6,
                'box': np.array([[10, 60], [100, 60], [100, 100], [10, 100]])
            }
        ]
        
        filtered_results = self.postprocessor.filter_by_confidence(results, 'commercial_name')
        
        # Should keep results above threshold
        self.assertLess(len(filtered_results), len(results))
    
    def test_merge_text_lines(self):
        """Test merging multiple text lines."""
        results = [
            {
                'text': 'Line 1',
                'confidence': 0.9,
                'box': np.array([[10, 10], [100, 10], [100, 20], [10, 20]])
            },
            {
                'text': 'Line 2',
                'confidence': 0.8,
                'box': np.array([[10, 30], [100, 30], [100, 40], [10, 40]])
            }
        ]
        
        merged = self.postprocessor.merge_text_lines(results)
        
        # Should return a dictionary with value and confidence
        self.assertIn('value', merged)
        self.assertIn('confidence', merged)
        
        # Value should contain both lines
        self.assertIn('Line 1', merged['value'])
        self.assertIn('Line 2', merged['value'])
    
    def test_extract_numeric_value(self):
        """Test extracting numeric value from text."""
        text = "Take 500 mg daily"
        
        numeric_value = self.postprocessor.extract_numeric_value(text)
        
        # Should extract 500
        self.assertEqual(numeric_value, 500)
    
    def test_extract_units(self):
        """Test extracting units from text."""
        text = "Take 500 mg daily"
        
        units = self.postprocessor.extract_units(text)
        
        # Should extract "mg"
        self.assertEqual(units, "mg")


class TestPackageProcessor(unittest.TestCase):
    """Test cases for PackageProcessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize package processor
        cls.package_processor = PackageProcessor(cls.config_path)
        
        # Create sample test image
        cls.test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(cls.test_image, "TestDrug 500mg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Create sample ROIs
        cls.test_rois = {
            ROIMapper.COMMERCIAL_NAME: [
                {
                    'image': cls.test_image.copy(),
                    'region': np.array([0, 0, 300, 100]),
                    'priority': 3
                }
            ],
            ROIMapper.DOSAGE: [
                {
                    'image': cls.test_image.copy(),
                    'region': np.array([0, 0, 300, 100]),
                    'priority': 2
                }
            ]
        }
    
    def test_initialization(self):
        """Test package processor initialization."""
        self.assertIsNotNone(self.package_processor)
        self.assertIsNotNone(self.package_processor.ocr_engine)
        self.assertIsNotNone(self.package_processor.preprocessor)
        self.assertIsNotNone(self.package_processor.postprocessor)
    
    def test_process(self):
        """Test processing package image with ROIs."""
        results = self.package_processor.process(self.test_image, self.test_rois)
        
        # Should return results for both fields
        self.assertIn(ROIMapper.COMMERCIAL_NAME, results)
        self.assertIn(ROIMapper.DOSAGE, results)
        
        # Each field should have value and confidence
        for field in [ROIMapper.COMMERCIAL_NAME, ROIMapper.DOSAGE]:
            self.assertIn('value', results[field])
            self.assertIn('confidence', results[field])
    
    def test_process_field(self):
        """Test processing individual field."""
        field_results = self.package_processor._process_field(
            self.test_image,
            self.test_rois[ROIMapper.COMMERCIAL_NAME],
            ROIMapper.COMMERCIAL_NAME
        )
        
        # Should return a dictionary with value and confidence
        self.assertIn('value', field_results)
        self.assertIn('confidence', field_results)
    
    def test_enhance_dosage_region(self):
        """Test enhancement of dosage region."""
        enhanced = self.package_processor._enhance_dosage_region(self.test_image)
        
        # Should return an image with same dimensions
        self.assertEqual(enhanced.shape, self.test_image.shape)
    
    def test_enhance_brand_region(self):
        """Test enhancement of brand name region."""
        enhanced = self.package_processor._enhance_brand_region(self.test_image)
        
        # Should return an image with same dimensions
        self.assertEqual(enhanced.shape, self.test_image.shape)


class TestSheetProcessor(unittest.TestCase):
    """Test cases for SheetProcessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize sheet processor
        cls.sheet_processor = SheetProcessor(cls.config_path)
        
        # Create sample test image with simulated text lines
        cls.test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        
        # Add multiple lines of text
        cv2.putText(cls.test_image, "Drug Name: TestDrug", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(cls.test_image, "Active: Acetaminophen", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(cls.test_image, "Manufacturer: Pharma Inc", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(cls.test_image, "Dosage: 500 mg", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Create sample ROIs
        cls.test_rois = {
            ROIMapper.COMMERCIAL_NAME: [
                {
                    'image': cls.test_image[10:40, :, :],
                    'region': np.array([0, 10, 300, 40]),
                    'priority': 3
                }
            ],
            ROIMapper.SCIENTIFIC_NAME: [
                {
                    'image': cls.test_image[40:70, :, :],
                    'region': np.array([0, 40, 300, 70]),
                    'priority': 2
                }
            ],
            ROIMapper.MANUFACTURER: [
                {
                    'image': cls.test_image[70:100, :, :],
                    'region': np.array([0, 70, 300, 100]),
                    'priority': 1
                }
            ],
            ROIMapper.DOSAGE: [
                {
                    'image': cls.test_image[100:130, :, :],
                    'region': np.array([0, 100, 300, 130]),
                    'priority': 2
                }
            ]
        }
    
    def test_initialization(self):
        """Test sheet processor initialization."""
        self.assertIsNotNone(self.sheet_processor)
        self.assertIsNotNone(self.sheet_processor.ocr_engine)
        self.assertIsNotNone(self.sheet_processor.preprocessor)
        self.assertIsNotNone(self.sheet_processor.postprocessor)
    
    def test_process(self):
        """Test processing sheet image with ROIs."""
        results = self.sheet_processor.process(self.test_image, self.test_rois)
        
        # Should return results for all fields
        for field in [ROIMapper.COMMERCIAL_NAME, ROIMapper.SCIENTIFIC_NAME, 
                     ROIMapper.MANUFACTURER, ROIMapper.DOSAGE]:
            self.assertIn(field, results)
            
            # Each field should have value and confidence
            self.assertIn('value', results[field])
            self.assertIn('confidence', results[field])
    
    def test_process_field(self):
        """Test processing individual field."""
        field_results = self.sheet_processor._process_field(
            self.test_image,
            self.test_rois[ROIMapper.COMMERCIAL_NAME],
            ROIMapper.COMMERCIAL_NAME
        )
        
        # Should return a dictionary with value and confidence
        self.assertIn('value', field_results)
        self.assertIn('confidence', field_results)
    
    def test_apply_specialized_preprocessing(self):
        """Test specialized preprocessing for different field types."""
        # Test for commercial name
        commercial_image = self.test_rois[ROIMapper.COMMERCIAL_NAME][0]['image']
        processed = self.sheet_processor._apply_specialized_preprocessing(
            commercial_image,
            ROIMapper.COMMERCIAL_NAME
        )
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape, commercial_image.shape)
        
        # Test for dosage
        dosage_image = self.test_rois[ROIMapper.DOSAGE][0]['image']
        processed = self.sheet_processor._apply_specialized_preprocessing(
            dosage_image,
            ROIMapper.DOSAGE
        )
        
        # Should return an image with same dimensions
        self.assertEqual(processed.shape, dosage_image.shape)
    
    def test_process_sections(self):
        """Test section-based processing."""
        results = self.sheet_processor.process_sections(self.test_image, None)
        
        # Should return some results
        self.assertGreater(len(results), 0)
    
    def test_header_text_enhancement(self):
        """Test header text enhancement."""
        enhanced = self.sheet_processor._enhance_header_text(self.test_image[10:40, :, :])
        
        # Should return an image with same dimensions
        self.assertEqual(enhanced.shape, self.test_image[10:40, :, :].shape)
    
    def test_extract_field_from_section_text(self):
        """Test extracting field from section text."""
        text_results = [
            {
                'text': 'Drug Name: TestDrug',
                'confidence': 0.9
            },
            {
                'text': 'Active: Acetaminophen',
                'confidence': 0.85
            }
        ]
        
        # Extract commercial name
        commercial_name = self.sheet_processor._extract_field_from_section_text(
            text_results,
            ROIMapper.COMMERCIAL_NAME
        )
        
        # Should extract "TestDrug"
        self.assertIsNotNone(commercial_name)
        self.assertIn("TestDrug", commercial_name)
        
        # Extract scientific name
        scientific_name = self.sheet_processor._extract_field_from_section_text(
            text_results,
            ROIMapper.SCIENTIFIC_NAME
        )
        
        # Should extract "Acetaminophen"
        self.assertIsNotNone(scientific_name)
        self.assertIn("Acetaminophen", scientific_name)


if __name__ == '__main__':
    unittest.main()
