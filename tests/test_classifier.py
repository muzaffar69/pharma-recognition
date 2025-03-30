"""
Unit tests for document classifier module.
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
from src.classification.document_classifier import DocumentClassifier
from src.classification.feature_extractor import FeatureExtractor

class TestDocumentClassifier(unittest.TestCase):
    """Test cases for DocumentClassifier class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        
        # Create test directory if it doesn't exist
        os.makedirs("tests/test_data", exist_ok=True)
        
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize classifier
        cls.classifier = DocumentClassifier(cls.config_path)
        
        # Create sample test images
        cls.package_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cls.sheet_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Draw rectangle on package image to simulate blister pack
        cv2.rectangle(cls.package_image, (50, 50), (350, 250), (0, 0, 0), 2)
        
        # Draw lines on sheet image to simulate text
        for i in range(10):
            y = 100 + i * 50
            cv2.line(cls.sheet_image, (50, y), (550, y), (0, 0, 0), 2)
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.feature_extractor)
    
    def test_feature_extraction(self):
        """Test feature extraction for classification."""
        features = self.classifier.preprocess_image(self.package_image)
        self.assertIsNotNone(features)
        self.assertGreater(features.shape[1], 0)
    
    def test_package_classification(self):
        """Test classification of package image."""
        doc_type, confidence = self.classifier.classify(self.package_image)
        self.assertEqual(doc_type, DocumentClassifier.PACKAGE)
        self.assertGreaterEqual(confidence, 0.5)
    
    def test_sheet_classification(self):
        """Test classification of information sheet image."""
        doc_type, confidence = self.classifier.classify(self.sheet_image)
        self.assertEqual(doc_type, DocumentClassifier.INFORMATION_SHEET)
        self.assertGreaterEqual(confidence, 0.5)
    
    def test_batch_classification(self):
        """Test batch classification."""
        images = [self.package_image, self.sheet_image]
        results = self.classifier.classify_batch(images)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], DocumentClassifier.PACKAGE)
        self.assertEqual(results[1][0], DocumentClassifier.INFORMATION_SHEET)
    
    def test_text_density_analysis(self):
        """Test text density analysis."""
        package_score = self.classifier._analyze_text_density(self.package_image)
        sheet_score = self.classifier._analyze_text_density(self.sheet_image)
        
        self.assertLess(package_score, sheet_score)
    
    def test_layout_analysis(self):
        """Test layout analysis."""
        package_score = self.classifier._analyze_layout(self.package_image)
        sheet_score = self.classifier._analyze_layout(self.sheet_image)
        
        self.assertLess(package_score, sheet_score)
    
    def test_is_package(self):
        """Test is_package method."""
        self.assertTrue(self.classifier.is_package(self.package_image))
        self.assertFalse(self.classifier.is_package(self.sheet_image))
    
    def test_is_information_sheet(self):
        """Test is_information_sheet method."""
        self.assertTrue(self.classifier.is_information_sheet(self.sheet_image))
        self.assertFalse(self.classifier.is_information_sheet(self.package_image))
    
    def test_empty_image(self):
        """Test classification with empty image."""
        empty_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        doc_type, confidence = self.classifier.classify(empty_image)
        
        # Should default to package with low confidence
        self.assertEqual(doc_type, DocumentClassifier.PACKAGE)
        self.assertLess(confidence, 0.7)
    
    def test_performance(self):
        """Test classification performance."""
        import time
        
        start = time.time()
        self.classifier.classify(self.package_image)
        duration = (time.time() - start) * 1000  # Convert to ms
        
        # Classification should be fast (target: <50ms)
        self.assertLess(duration, 50)


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Initialize feature extractors for different types
        cls.hog_extractor = FeatureExtractor(feature_type='hog')
        cls.orb_extractor = FeatureExtractor(feature_type='orb')
        cls.sift_extractor = FeatureExtractor(feature_type='sift')
        cls.custom_extractor = FeatureExtractor(feature_type='custom')
        
        # Create sample image
        cls.test_image = np.ones((300, 400), dtype=np.uint8) * 255
        cv2.rectangle(cls.test_image, (50, 50), (350, 250), 0, 2)
    
    def test_hog_extraction(self):
        """Test HOG feature extraction."""
        features = self.hog_extractor.extract(self.test_image)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], self.hog_extractor.feature_dim)
    
    def test_orb_extraction(self):
        """Test ORB feature extraction."""
        features = self.orb_extractor.extract(self.test_image)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], self.orb_extractor.feature_dim)
    
    def test_sift_extraction(self):
        """Test SIFT feature extraction."""
        features = self.sift_extractor.extract(self.test_image)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], self.sift_extractor.feature_dim)
    
    def test_custom_extraction(self):
        """Test custom feature extraction."""
        features = self.custom_extractor.extract(self.test_image)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], self.custom_extractor.feature_dim)
    
    def test_keypoint_extraction(self):
        """Test keypoint extraction."""
        keypoints, descriptors = self.orb_extractor.extract_keypoints(self.test_image)
        self.assertIsNotNone(keypoints)
        self.assertIsNotNone(descriptors)
        self.assertGreater(len(keypoints), 0)
    
    def test_empty_image(self):
        """Test feature extraction with empty image."""
        empty_image = np.ones((10, 10), dtype=np.uint8) * 255
        features = self.hog_extractor.extract(empty_image)
        self.assertIsNotNone(features)
    
    def test_performance(self):
        """Test extraction performance."""
        import time
        
        start = time.time()
        self.hog_extractor.extract(self.test_image)
        duration = (time.time() - start) * 1000  # Convert to ms
        
        # Feature extraction should be fast
        self.assertLess(duration, 50)


if __name__ == '__main__':
    unittest.main()
