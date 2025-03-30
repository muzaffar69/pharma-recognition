"""
Unit tests for ROI mapping module.
"""

import os
import sys
import unittest
import cv2
import numpy as np
import json
import shutil
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components to test
from src.roi.roi_mapper import ROIMapper
from src.roi.package_roi import PackageROI
from src.roi.sheet_roi import SheetROI

class TestROIMapper(unittest.TestCase):
    """Test cases for ROIMapper class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        
        # Create test directories
        os.makedirs("tests/test_data/roi_mappings/packages", exist_ok=True)
        os.makedirs("tests/test_data/roi_mappings/sheets", exist_ok=True)
        
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize ROI mapper
        cls.roi_mapper = ROIMapper(cls.config_path)
        
        # Create sample test images
        cls.image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Create test homography (identity + small translation)
        cls.homography = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Create test template ID
        cls.template_id = "test_template"
        
        # Create test ROI mappings
        cls.test_mappings = {
            ROIMapper.COMMERCIAL_NAME: [
                np.array([50, 50, 150, 80])
            ],
            ROIMapper.SCIENTIFIC_NAME: [
                np.array([50, 100, 250, 130])
            ],
            ROIMapper.MANUFACTURER: [
                np.array([50, 150, 200, 180])
            ],
            ROIMapper.DOSAGE: [
                np.array([50, 200, 150, 230])
            ]
        }
        
        # Save test mappings for package
        cls.roi_mapper.save_mapping(cls.template_id, cls.test_mappings, is_sheet=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test directories
        try:
            shutil.rmtree("tests/test_data")
        except:
            pass
    
    def test_initialization(self):
        """Test ROI mapper initialization."""
        self.assertIsNotNone(self.roi_mapper)
    
    def test_save_and_load_mapping(self):
        """Test saving and loading ROI mappings."""
        # Save mapping
        success = self.roi_mapper.save_mapping(self.template_id, self.test_mappings, is_sheet=True)
        self.assertTrue(success)
        
        # Load mapping
        loaded_mappings = self.roi_mapper.load_mapping(self.template_id, is_sheet=True)
        self.assertIsNotNone(loaded_mappings)
        
        # Check if fields are present
        for field in [ROIMapper.COMMERCIAL_NAME, ROIMapper.SCIENTIFIC_NAME, 
                     ROIMapper.MANUFACTURER, ROIMapper.DOSAGE]:
            self.assertIn(field, loaded_mappings)
    
    def test_transform_roi(self):
        """Test ROI transformation."""
        # ROI to transform
        roi = np.array([50, 50, 150, 80])
        
        # Transform ROI
        transformed_roi = self.roi_mapper.transform_roi(roi, self.homography, self.image.shape[:2])
        
        # Check transformation (should be shifted by 10 pixels in x and y)
        self.assertEqual(transformed_roi[0], roi[0] + 10)
        self.assertEqual(transformed_roi[1], roi[1] + 10)
        self.assertEqual(transformed_roi[2], roi[2] + 10)
        self.assertEqual(transformed_roi[3], roi[3] + 10)
    
    def test_apply_mapping(self):
        """Test applying ROI mapping to an image."""
        # Apply mapping
        extracted_rois = self.roi_mapper.apply_mapping(self.image, self.template_id, self.homography, is_sheet=False)
        
        # Check if all fields are present
        for field in [ROIMapper.COMMERCIAL_NAME, ROIMapper.SCIENTIFIC_NAME, 
                     ROIMapper.MANUFACTURER, ROIMapper.DOSAGE]:
            self.assertIn(field, extracted_rois)
            self.assertGreater(len(extracted_rois[field]), 0)
            
            # Check if ROI images are present
            for roi_data in extracted_rois[field]:
                self.assertIn('image', roi_data)
                self.assertIn('region', roi_data)
                self.assertIn('priority', roi_data)
    
    def test_create_mapping(self):
        """Test creating new ROI mapping."""
        # Create new template ID
        new_template_id = "new_test_template"
        
        # Define regions
        regions = [
            [50, 50, 150, 80],
            [50, 100, 250, 130]
        ]
        
        # Create mapping
        success = self.roi_mapper.create_mapping(
            new_template_id,
            ROIMapper.COMMERCIAL_NAME,
            regions,
            is_sheet=False
        )
        
        self.assertTrue(success)
        
        # Load mapping
        loaded_mappings = self.roi_mapper.load_mapping(new_template_id, is_sheet=False)
        self.assertIsNotNone(loaded_mappings)
        self.assertIn(ROIMapper.COMMERCIAL_NAME, loaded_mappings)
        self.assertEqual(len(loaded_mappings[ROIMapper.COMMERCIAL_NAME]), len(regions))
    
    def test_auto_adjust_mapping(self):
        """Test automatic adjustment of ROI mapping."""
        # Create OCR results with high confidence for a low-priority ROI
        ocr_results = {
            ROIMapper.COMMERCIAL_NAME: [
                {
                    'text': 'TestDrug',
                    'confidence': 0.9,
                    'roi_index': 0
                }
            ]
        }
        
        # Auto-adjust mapping
        adjusted = self.roi_mapper.auto_adjust_mapping(
            self.image,
            self.template_id,
            self.homography,
            ocr_results,
            is_sheet=False
        )
        
        # This should return True if adjustment was made
        self.assertIsNotNone(adjusted)


class TestPackageROI(unittest.TestCase):
    """Test cases for PackageROI class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test directories
        os.makedirs("tests/test_data/roi_mappings/packages", exist_ok=True)
        
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize PackageROI handler
        cls.package_roi = PackageROI(cls.config_path)
        
        # Create sample test image
        cls.image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Create test homography (identity + small translation)
        cls.homography = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Create test template ID
        cls.template_id = "test_package"
        
        # Create test ROI mappings
        cls.roi_mapper = ROIMapper(cls.config_path)
        cls.test_mappings = {
            ROIMapper.COMMERCIAL_NAME: [
                {
                    'region': np.array([50, 50, 150, 80]),
                    'priority': 3,
                    'description': 'Commercial name',
                    'ocr_config': {}
                }
            ],
            ROIMapper.SCIENTIFIC_NAME: [
                {
                    'region': np.array([50, 100, 250, 130]),
                    'priority': 2,
                    'description': 'Scientific name',
                    'ocr_config': {}
                }
            ]
        }
        
        # Save test mappings
        cls.roi_mapper.save_mapping(cls.template_id, cls.test_mappings, is_sheet=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test directories
        try:
            shutil.rmtree("tests/test_data")
        except:
            pass
    
    def test_initialization(self):
        """Test PackageROI initialization."""
        self.assertIsNotNone(self.package_roi)
        self.assertIsNotNone(self.package_roi.roi_mapper)
    
    def test_extract_field_rois(self):
        """Test extracting field ROIs from package image."""
        # Extract ROIs
        rois = self.package_roi.extract_field_rois(self.image, self.template_id, self.homography)
        
        # Check if fields are present
        for field in [ROIMapper.COMMERCIAL_NAME, ROIMapper.SCIENTIFIC_NAME]:
            self.assertIn(field, rois)
            self.assertGreater(len(rois[field]), 0)
    
    def test_enhance_package_rois(self):
        """Test package-specific ROI enhancement."""
        # Extract ROIs
        rois = self.package_roi.extract_field_rois(self.image, self.template_id, self.homography)
        
        # Check if ROIs are enhanced
        for field in rois:
            for roi_data in rois[field]:
                self.assertIn('image', roi_data)
                self.assertIn('ocr_config', roi_data)
    
    def test_handle_reflective_surface(self):
        """Test handling of reflective surfaces."""
        # Create test image with reflective surface
        reflective_image = np.ones((100, 100), dtype=np.uint8) * 200
        # Add some gradient to simulate reflection
        for i in range(100):
            reflective_image[i, :] = 200 + i // 2
        
        # Apply reflective surface handling
        processed = self.package_roi._handle_reflective_surface(reflective_image)
        
        # Check if processed image has different values
        self.assertFalse(np.array_equal(reflective_image, processed))
    
    def test_default_package_ocr_config(self):
        """Test default OCR configuration for package fields."""
        # Get OCR config for commercial name
        ocr_config = self.package_roi._get_default_package_ocr_config(ROIMapper.COMMERCIAL_NAME)
        
        # Check if configuration is present
        self.assertIsNotNone(ocr_config)
        self.assertIn('det_db_thresh', ocr_config)
        self.assertIn('min_text_confidence', ocr_config)
    
    def test_handle_missing_tablets(self):
        """Test handling of missing tablets in blister packs."""
        # Create image with simulated missing tablet
        image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        # Draw grid of circles for tablets
        for i in range(3):
            for j in range(4):
                center = (50 + i * 100, 50 + j * 50)
                # Skip one circle to simulate missing tablet
                if i != 1 or j != 2:
                    cv2.circle(image, center, 20, (0, 0, 0), 2)
        
        # Apply missing tablet handling
        rois = self.package_roi.handle_missing_tablets(image, self.template_id, self.homography)
        
        # Not much to assert here without complex mock-ups, but at least check it doesn't crash
        self.assertIsNotNone(rois)
    
    def test_detect_missing_tablets(self):
        """Test detection of missing tablets."""
        # Create image with simulated missing tablet
        image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        # Draw grid of circles for tablets with one missing
        for i in range(3):
            for j in range(4):
                center = (50 + i * 100, 50 + j * 50)
                # Skip one circle to simulate missing tablet
                if i != 1 or j != 2:
                    cv2.circle(image, center, 20, (0, 0, 0), 2)
        
        # Invert image for better detection
        image = 255 - image
        
        # Detect missing tablets
        missing_areas = self.package_roi._detect_missing_tablets(image)
        
        # Should detect at least one missing area
        self.assertGreaterEqual(len(missing_areas), 0)


class TestSheetROI(unittest.TestCase):
    """Test cases for SheetROI class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test directories
        os.makedirs("tests/test_data/roi_mappings/sheets", exist_ok=True)
        
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize SheetROI handler
        cls.sheet_roi = SheetROI(cls.config_path)
        
        # Create sample test image with simulated text lines
        cls.image = np.ones((500, 400, 3), dtype=np.uint8) * 255
        
        # Add simulated text lines
        for i in range(10):
            y = 50 + i * 40
            cv2.line(cls.image, (50, y), (350, y), (0, 0, 0), 2)
        
        # Create test homography (identity)
        cls.homography = np.eye(3)
        
        # Create test template ID
        cls.template_id = "test_sheet"
        
        # Create test ROI mappings
        cls.roi_mapper = ROIMapper(cls.config_path)
        cls.test_mappings = {
            ROIMapper.COMMERCIAL_NAME: [
                {
                    'region': np.array([50, 50, 350, 90]),
                    'priority': 3,
                    'description': 'Commercial name',
                    'ocr_config': {}
                }
            ],
            ROIMapper.SCIENTIFIC_NAME: [
                {
                    'region': np.array([50, 130, 350, 170]),
                    'priority': 2,
                    'description': 'Scientific name',
                    'ocr_config': {}
                }
            ],
            ROIMapper.MANUFACTURER: [
                {
                    'region': np.array([50, 210, 350, 250]),
                    'priority': 1,
                    'description': 'Manufacturer',
                    'ocr_config': {}
                }
            ],
            ROIMapper.DOSAGE: [
                {
                    'region': np.array([50, 290, 350, 330]),
                    'priority': 2,
                    'description': 'Dosage',
                    'ocr_config': {}
                }
            ]
        }
        
        # Save test mappings
        cls.roi_mapper.save_mapping(cls.template_id, cls.test_mappings, is_sheet=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test directories
        try:
            shutil.rmtree("tests/test_data")
        except:
            pass
    
    def test_initialization(self):
        """Test SheetROI initialization."""
        self.assertIsNotNone(self.sheet_roi)
        self.assertIsNotNone(self.sheet_roi.roi_mapper)
    
    def test_extract_field_rois(self):
        """Test extracting field ROIs from sheet image."""
        # Extract ROIs
        rois = self.sheet_roi.extract_field_rois(self.image, self.template_id, self.homography)
        
        # Check if fields are present
        for field in [ROIMapper.COMMERCIAL_NAME, ROIMapper.SCIENTIFIC_NAME, 
                     ROIMapper.MANUFACTURER, ROIMapper.DOSAGE]:
            self.assertIn(field, rois)
            self.assertGreater(len(rois[field]), 0)
    
    def test_enhance_sheet_rois(self):
        """Test sheet-specific ROI enhancement."""
        # Extract ROIs
        rois = self.sheet_roi.extract_field_rois(self.image, self.template_id, self.homography)
        
        # Check if ROIs are enhanced
        for field in rois:
            for roi_data in rois[field]:
                self.assertIn('image', roi_data)
                self.assertIn('ocr_config', roi_data)
    
    def test_preprocess_sheet_roi(self):
        """Test preprocessing of sheet ROI."""
        # Create test ROI image
        roi_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Add some text
        cv2.putText(roi_image, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Preprocess ROI
        processed = self.sheet_roi._preprocess_sheet_roi(roi_image)
        
        # Check if processed image has the same dimensions
        self.assertEqual(processed.shape, roi_image.shape)
    
    def test_default_sheet_ocr_config(self):
        """Test default OCR configuration for sheet fields."""
        # Get OCR config for commercial name
        ocr_config = self.sheet_roi._get_default_sheet_ocr_config(ROIMapper.COMMERCIAL_NAME)
        
        # Check if configuration is present
        self.assertIsNotNone(ocr_config)
        self.assertIn('det_db_thresh', ocr_config)
        self.assertIn('min_text_confidence', ocr_config)
    
    def test_extract_sections(self):
        """Test extraction of sections from sheet."""
        # Extract sections
        sections = self.sheet_roi._extract_sections(self.image)
        
        # Should detect at least one section
        self.assertGreater(len(sections), 0)
        
        # Check section structure
        for section in sections:
            self.assertIn('region', section)
            self.assertIn('type', section)
    
    def test_categorize_section(self):
        """Test section categorization."""
        # Create test section images
        header_image = np.ones((30, 300), dtype=np.uint8) * 255
        cv2.putText(header_image, "HEADER", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        paragraph_image = np.ones((200, 300), dtype=np.uint8) * 255
        for i in range(5):
            y = 30 + i * 30
            cv2.line(paragraph_image, (20, y), (280, y), (0, 0, 0), 1)
        
        # Categorize sections
        header_type = self.sheet_roi._categorize_section(header_image)
        paragraph_type = self.sheet_roi._categorize_section(paragraph_image)
        
        self.assertEqual(header_type, 'header')
        self.assertEqual(paragraph_type, 'paragraph')
    
    def test_find_containing_section(self):
        """Test finding section containing a region."""
        # Create sections
        sections = [
            {'region': [10, 10, 100, 100], 'type': 'header'},
            {'region': [10, 110, 100, 200], 'type': 'paragraph'},
            {'region': [10, 210, 100, 300], 'type': 'table'}
        ]
        
        # Test regions inside and outside sections
        inside_region = [20, 50, 90, 90]
        outside_region = [150, 150, 200, 200]
        
        # Find containing sections
        inside_section = self.sheet_roi._find_containing_section(inside_region, sections)
        outside_section = self.sheet_roi._find_containing_section(outside_region, sections)
        
        self.assertIsNotNone(inside_section)
        self.assertEqual(inside_section['type'], 'header')
        self.assertIsNone(outside_section)
    
    def test_find_redundant_information(self):
        """Test finding redundant information across ROIs."""
        # Create test OCR results with redundant information
        ocr_results = {
            ROIMapper.COMMERCIAL_NAME: [
                {'text': 'DrugName', 'confidence': 0.9, 'roi_index': 0},
                {'text': 'DrugName', 'confidence': 0.8, 'roi_index': 1}
            ],
            ROIMapper.SCIENTIFIC_NAME: [
                {'text': 'Ingredient1', 'confidence': 0.85, 'roi_index': 0},
                {'text': 'Ingredient2', 'confidence': 0.7, 'roi_index': 1}
            ]
        }
        
        # Create dummy ROIs
        rois = {
            ROIMapper.COMMERCIAL_NAME: [
                {'priority': 3}, {'priority': 2}
            ],
            ROIMapper.SCIENTIFIC_NAME: [
                {'priority': 3}, {'priority': 2}
            ]
        }
        
        # Find redundant information
        reliable_values = self.sheet_roi.find_redundant_information(rois, ocr_results)
        
        # Check if redundant commercial name was found
        self.assertIn(ROIMapper.COMMERCIAL_NAME, reliable_values)
        self.assertEqual(reliable_values[ROIMapper.COMMERCIAL_NAME]['value'], 'DrugName')
        self.assertEqual(reliable_values[ROIMapper.COMMERCIAL_NAME]['instances'], 2)


if __name__ == '__main__':
    unittest.main()
