"""
Integration tests for the complete pharmaceutical recognition pipeline.
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

# Import main system
from src.main import PharmaceuticalRecognitionSystem

class TestPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        
        # Create test directories
        os.makedirs("tests/test_data/templates/packages", exist_ok=True)
        os.makedirs("tests/test_data/templates/sheets", exist_ok=True)
        os.makedirs("tests/test_data/roi_mappings/packages", exist_ok=True)
        os.makedirs("tests/test_data/roi_mappings/sheets", exist_ok=True)
        os.makedirs("tests/test_output", exist_ok=True)
        
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize the system
        cls.system = PharmaceuticalRecognitionSystem(cls.config_path)
        
        # Create sample test images
        cls.package_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cls.sheet_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Simulate package with brand name and dosage
        cv2.rectangle(cls.package_image, (50, 50), (350, 250), (0, 0, 0), 2)
        cv2.putText(cls.package_image, "TestDrug", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(cls.package_image, "500mg", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Simulate information sheet with multiple text sections
        for i in range(10):
            y = 100 + i * 50
            cv2.line(cls.sheet_image, (50, y), (550, y), (0, 0, 0), 2)
        
        cv2.putText(cls.sheet_image, "Drug Name: TestDrug", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(cls.sheet_image, "Active: Acetaminophen", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(cls.sheet_image, "Manufacturer: Pharma Inc", (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(cls.sheet_image, "Dosage: 500 mg", (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save test images
        cv2.imwrite("tests/test_data/package.jpg", cls.package_image)
        cv2.imwrite("tests/test_data/sheet.jpg", cls.sheet_image)
        
        # Add template for package
        cls.package_template_id = "test_package"
        cls.system.add_template(cls.package_image, cls.package_template_id, is_sheet=False)
        
        # Add template for sheet
        cls.sheet_template_id = "test_sheet"
        cls.system.add_template(cls.sheet_image, cls.sheet_template_id, is_sheet=True)
        
        # Create ROI mappings for package
        roi_mapper = cls.system.package_roi.roi_mapper
        package_mappings = {
            roi_mapper.COMMERCIAL_NAME: [
                {
                    'region': np.array([90, 80, 250, 120]),
                    'priority': 3,
                    'description': 'Commercial name',
                    'ocr_config': {}
                }
            ],
            roi_mapper.DOSAGE: [
                {
                    'region': np.array([90, 130, 200, 170]),
                    'priority': 2,
                    'description': 'Dosage',
                    'ocr_config': {}
                }
            ]
        }
        roi_mapper.save_mapping(cls.package_template_id, package_mappings, is_sheet=False)
        
        # Create ROI mappings for sheet
        sheet_mappings = {
            roi_mapper.COMMERCIAL_NAME: [
                {
                    'region': np.array([70, 110, 350, 150]),
                    'priority': 3,
                    'description': 'Commercial name',
                    'ocr_config': {}
                }
            ],
            roi_mapper.SCIENTIFIC_NAME: [
                {
                    'region': np.array([70, 160, 400, 200]),
                    'priority': 2,
                    'description': 'Scientific name',
                    'ocr_config': {}
                }
            ],
            roi_mapper.MANUFACTURER: [
                {
                    'region': np.array([70, 210, 400, 250]),
                    'priority': 1,
                    'description': 'Manufacturer',
                    'ocr_config': {}
                }
            ],
            roi_mapper.DOSAGE: [
                {
                    'region': np.array([70, 260, 300, 300]),
                    'priority': 2,
                    'description': 'Dosage',
                    'ocr_config': {}
                }
            ]
        }
        roi_mapper.save_mapping(cls.sheet_template_id, sheet_mappings, is_sheet=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test directories
        try:
            shutil.rmtree("tests/test_data")
            shutil.rmtree("tests/test_output")
        except:
            pass
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertIsNotNone(self.system.document_classifier)
        self.assertIsNotNone(self.system.feature_matcher)
        self.assertIsNotNone(self.system.package_roi)
        self.assertIsNotNone(self.system.sheet_roi)
        self.assertIsNotNone(self.system.package_processor)
        self.assertIsNotNone(self.system.sheet_processor)
    
    def test_process_package_image(self):
        """Test processing a package image."""
        # Process the package image
        results = self.system.process(self.package_image)
        
        # Should classify correctly
        self.assertFalse(results['is_sheet'])
        
        # Should match the template
        self.assertEqual(results['template_id'], self.package_template_id)
        
        # Should have high confidence
        self.assertGreaterEqual(results['template_confidence'], 0.7)
        
        # Should extract commercial name and dosage
        self.assertIn('commercial_name', results)
        self.assertIn('dosage', results)
        
        # Check commercial name value
        commercial_name = results['commercial_name']['value']
        self.assertIsNotNone(commercial_name)
        self.assertIn('TESTDRUG', commercial_name.upper())
        
        # Check dosage value
        dosage = results['dosage']['value']
        self.assertIsNotNone(dosage)
        self.assertIn('500', dosage)
    
    def test_process_sheet_image(self):
        """Test processing a sheet image."""
        # Process the sheet image
        results = self.system.process(self.sheet_image)
        
        # Should classify correctly
        self.assertTrue(results['is_sheet'])
        
        # Should match the template
        self.assertEqual(results['template_id'], self.sheet_template_id)
        
        # Should have high confidence
        self.assertGreaterEqual(results['template_confidence'], 0.7)
        
        # Should extract all fields
        for field in ['commercial_name', 'scientific_name', 'manufacturer', 'dosage']:
            self.assertIn(field, results)
            self.assertIn('value', results[field])
            self.assertIn('confidence', results[field])
        
        # Check field values
        self.assertIn('TestDrug', results['commercial_name']['value'])
        self.assertIn('Acetaminophen', results['scientific_name']['value'])
        self.assertIn('Pharma', results['manufacturer']['value'])
        self.assertIn('500', results['dosage']['value'])
    
    def test_process_image_file(self):
        """Test processing an image file."""
        # Process the package image file
        output_path = "tests/test_output/package_results.json"
        results = self.system.process_image("tests/test_data/package.jpg", output_path)
        
        # Check that results file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load results from file
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        # Should have the same fields
        self.assertEqual(results["is_sheet"], loaded_results["is_sheet"])
        self.assertEqual(results["template_id"], loaded_results["template_id"])
        
        # Check processing time
        self.assertIn("processing_time_ms", results)
        self.assertLess(results["processing_time_ms"], 500)  # Should be under 500ms
    
    def test_process_batch(self):
        """Test batch processing."""
        # Process a batch of images
        output_dir = "tests/test_output"
        batch_results = self.system.process_batch("tests/test_data", output_dir)
        
        # Should process both images
        self.assertEqual(batch_results["total_images"], 2)
        self.assertEqual(batch_results["successful"], 2)
        self.assertEqual(batch_results["failed"], 0)
        
        # Check that output files were created
        self.assertTrue(os.path.exists(os.path.join(output_dir, "package_results.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "sheet_results.json")))
    
    def test_end_to_end_performance(self):
        """Test end-to-end performance."""
        import time
        
        # Measure processing time for package
        start_time = time.time()
        self.system.process(self.package_image)
        package_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Measure processing time for sheet
        start_time = time.time()
        self.system.process(self.sheet_image)
        sheet_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Both should be under 500ms
        self.assertLess(package_time, 500)
        self.assertLess(sheet_time, 500)
        
        # Log performance
        logger.info(f"Package processing time: {package_time:.2f}ms")
        logger.info(f"Sheet processing time: {sheet_time:.2f}ms")
    
    def test_add_and_remove_template(self):
        """Test adding and removing templates."""
        # Create a new test image
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_image, (100, 100), (300, 200), (0, 0, 0), 2)
        cv2.putText(test_image, "NewDrug", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add as a template
        template_id = "new_template"
        success = self.system.add_template(test_image, template_id, is_sheet=False)
        self.assertTrue(success)
        
        # Process the image
        results = self.system.process(test_image)
        
        # Should match the new template
        self.assertEqual(results['template_id'], template_id)
        
        # Remove the template
        success = self.system.feature_matcher.remove_template(template_id, is_sheet=False)
        self.assertTrue(success)
    
    def test_component_interaction(self):
        """Test interaction between system components."""
        # Test classification + template matching interaction
        doc_type, _ = self.system.document_classifier.classify(self.package_image)
        is_sheet = doc_type == self.system.document_classifier.INFORMATION_SHEET
        
        template_id, _, confidence, homography = self.system.feature_matcher.identify_template(
            self.package_image, is_sheet
        )
        
        # Should match the package template
        self.assertEqual(template_id, self.package_template_id)
        self.assertGreaterEqual(confidence, 0.7)
        
        # Test template matching + ROI mapping interaction
        rois = self.system.package_roi.extract_field_rois(
            self.package_image, template_id, homography
        )
        
        # Should have ROIs for commercial name and dosage
        self.assertIn('commercial_name', rois)
        self.assertIn('dosage', rois)
        
        # Test ROI mapping + OCR interaction
        extraction_results = self.system.package_processor.process(
            self.package_image, rois
        )
        
        # Should extract commercial name and dosage
        self.assertIn('commercial_name', extraction_results)
        self.assertIn('dosage', extraction_results)
        self.assertIn('value', extraction_results['commercial_name'])
        self.assertIn('confidence', extraction_results['commercial_name'])
    
    def test_benchmark(self):
        """Test benchmark functionality."""
        # Run benchmark with minimal iterations
        benchmark_results = self.system.benchmark("tests/test_data/package.jpg", iterations=2)
        
        # Check benchmark results
        self.assertIn("iterations", benchmark_results)
        self.assertIn("image_path", benchmark_results)
        self.assertIn("results", benchmark_results)
        self.assertIn("statistics", benchmark_results)
        
        # Check component timing statistics
        component_stats = benchmark_results["statistics"]
        for component in ['document_classification', 'template_matching', 'roi_mapping', 'ocr_recognition']:
            self.assertIn(component, component_stats)
            self.assertIn('mean', component_stats[component])
            self.assertIn('min', component_stats[component])
            self.assertIn('max', component_stats[component])


if __name__ == '__main__':
    unittest.main()
