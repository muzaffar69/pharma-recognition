"""
Unit tests for template matching module.
"""

import os
import sys
import unittest
import cv2
import numpy as np
import shutil
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components to test
from src.template_matching.feature_matcher import FeatureMatcher
from src.template_matching.template_indexer import TemplateIndexer
from src.template_matching.transformation import TransformationEstimator
from src.classification.feature_extractor import FeatureExtractor

class TestFeatureMatcher(unittest.TestCase):
    """Test cases for FeatureMatcher class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        
        # Create test directories
        os.makedirs("tests/test_data/templates/packages", exist_ok=True)
        os.makedirs("tests/test_data/templates/sheets", exist_ok=True)
        
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize feature matcher
        cls.feature_matcher = FeatureMatcher(cls.config_path)
        
        # Create sample test images
        cls.template_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cls.query_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Draw rectangle on images
        cv2.rectangle(cls.template_image, (50, 50), (350, 250), (0, 0, 0), 2)
        cv2.rectangle(cls.query_image, (60, 60), (360, 260), (0, 0, 0), 2)  # Slightly shifted
        
        # Add template to the database
        cls.template_id = "test_template"
        cls.feature_matcher.add_template(cls.template_image, cls.template_id, is_sheet=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test directories
        try:
            shutil.rmtree("tests/test_data")
        except:
            pass
    
    def test_initialization(self):
        """Test feature matcher initialization."""
        self.assertIsNotNone(self.feature_matcher)
        self.assertIsNotNone(self.feature_matcher.feature_extractor)
        self.assertIsNotNone(self.feature_matcher.package_indexer)
        self.assertIsNotNone(self.feature_matcher.sheet_indexer)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        image, keypoints, descriptors = self.feature_matcher.preprocess_image(self.query_image)
        self.assertIsNotNone(image)
        self.assertIsNotNone(keypoints)
        self.assertIsNotNone(descriptors)
        self.assertGreater(len(keypoints), 0)
    
    def test_match_template(self):
        """Test template matching."""
        template_id, confidence, homography = self.feature_matcher.match_template(
            self.query_image, 
            template_id=self.template_id, 
            is_sheet=False
        )
        
        self.assertEqual(template_id, self.template_id)
        self.assertGreaterEqual(confidence, 0.5)
        self.assertIsNotNone(homography)
    
    def test_identify_template(self):
        """Test template identification."""
        template_id, template_type, confidence, homography = self.feature_matcher.identify_template(
            self.query_image
        )
        
        self.assertEqual(template_id, self.template_id)
        self.assertEqual(template_type, 'package')
        self.assertGreaterEqual(confidence, 0.5)
        self.assertIsNotNone(homography)
    
    def test_add_and_remove_template(self):
        """Test adding and removing templates."""
        # Add new template
        new_template_id = "test_template_2"
        success = self.feature_matcher.add_template(self.template_image, new_template_id, is_sheet=True)
        self.assertTrue(success)
        
        # Match against new template
        template_id, confidence, homography = self.feature_matcher.match_template(
            self.query_image, 
            template_id=new_template_id, 
            is_sheet=True
        )
        
        self.assertEqual(template_id, new_template_id)
        
        # Remove template
        success = self.feature_matcher.remove_template(new_template_id, is_sheet=True)
        self.assertTrue(success)
        
        # Try to match against removed template
        template_id, confidence, homography = self.feature_matcher.match_template(
            self.query_image, 
            template_id=new_template_id, 
            is_sheet=True
        )
        
        self.assertIsNone(template_id)
    
    def test_template_not_found(self):
        """Test behavior when template is not found."""
        # Create a completely different image
        different_image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(different_image, (200, 150), 100, (255, 255, 255), -1)
        
        template_id, template_type, confidence, homography = self.feature_matcher.identify_template(
            different_image
        )
        
        self.assertIsNone(template_id)
        self.assertLess(confidence, 0.5)
    
    def test_performance(self):
        """Test matching performance."""
        import time
        
        start = time.time()
        self.feature_matcher.identify_template(self.query_image)
        duration = (time.time() - start) * 1000  # Convert to ms
        
        # Template matching should be fast (target: <100ms)
        self.assertLess(duration, 100)


class TestTemplateIndexer(unittest.TestCase):
    """Test cases for TemplateIndexer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test directories
        os.makedirs("tests/test_data/templates", exist_ok=True)
        
        # Initialize feature extractor
        cls.feature_extractor = FeatureExtractor(feature_type='orb')
        
        # Initialize template indexers with different storage types
        cls.lmdb_indexer = TemplateIndexer(cls.feature_extractor, "tests/test_data/templates/lmdb", "lmdb")
        cls.pickle_indexer = TemplateIndexer(cls.feature_extractor, "tests/test_data/templates/pickle", "pickle")
        cls.memory_indexer = TemplateIndexer(cls.feature_extractor, "tests/test_data/templates/memory", "memory")
        
        # Create sample test image
        cls.test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(cls.test_image, (50, 50), (350, 250), (0, 0, 0), 2)
        
        # Extract keypoints and descriptors
        cls.keypoints, cls.descriptors = cls.feature_extractor.extract_keypoints(cls.test_image)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test directories
        try:
            shutil.rmtree("tests/test_data")
        except:
            pass
    
    def test_lmdb_indexer(self):
        """Test LMDB-based template indexer."""
        # Add template
        template_id = "test_lmdb"
        success = self.lmdb_indexer.add_template(template_id, self.keypoints, self.descriptors, self.test_image)
        self.assertTrue(success)
        
        # Get template
        template = self.lmdb_indexer.get_template(template_id)
        self.assertIsNotNone(template)
        self.assertIn('keypoints', template)
        self.assertIn('descriptors', template)
        self.assertIn('image_path', template)
        
        # Get all template IDs
        template_ids = self.lmdb_indexer.get_all_template_ids()
        self.assertIn(template_id, template_ids)
        
        # Count templates
        count = self.lmdb_indexer.count()
        self.assertEqual(count, 1)
        
        # Remove template
        success = self.lmdb_indexer.remove_template(template_id)
        self.assertTrue(success)
        
        # Verify template was removed
        template = self.lmdb_indexer.get_template(template_id)
        self.assertIsNone(template)
    
    def test_pickle_indexer(self):
        """Test pickle-based template indexer."""
        # Add template
        template_id = "test_pickle"
        success = self.pickle_indexer.add_template(template_id, self.keypoints, self.descriptors, self.test_image)
        self.assertTrue(success)
        
        # Get template
        template = self.pickle_indexer.get_template(template_id)
        self.assertIsNotNone(template)
        
        # Remove template
        success = self.pickle_indexer.remove_template(template_id)
        self.assertTrue(success)
    
    def test_memory_indexer(self):
        """Test memory-based template indexer."""
        # Add template
        template_id = "test_memory"
        success = self.memory_indexer.add_template(template_id, self.keypoints, self.descriptors, self.test_image)
        self.assertTrue(success)
        
        # Get template
        template = self.memory_indexer.get_template(template_id)
        self.assertIsNotNone(template)
        
        # Clear all templates
        success = self.memory_indexer.clear()
        self.assertTrue(success)
        
        # Verify all templates were removed
        count = self.memory_indexer.count()
        self.assertEqual(count, 0)


class TestTransformationEstimator(unittest.TestCase):
    """Test cases for TransformationEstimator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Path to configuration file
        cls.config_path = 'config/system_config.json'
        
        # Initialize transformation estimator
        cls.transformation = TransformationEstimator(cls.config_path)
        
        # Initialize feature extractor
        cls.feature_extractor = FeatureExtractor(feature_type='orb')
        
        # Create sample test images
        cls.src_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cls.dst_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Draw rectangle on images
        cv2.rectangle(cls.src_image, (50, 50), (350, 250), (0, 0, 0), 2)
        cv2.rectangle(cls.dst_image, (60, 60), (360, 260), (0, 0, 0), 2)  # Slightly shifted
        
        # Extract keypoints and descriptors
        cls.src_keypoints, cls.src_descriptors = cls.feature_extractor.extract_keypoints(cls.src_image)
        cls.dst_keypoints, cls.dst_descriptors = cls.feature_extractor.extract_keypoints(cls.dst_image)
        
        # Create matcher
        cls.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        cls.matches = cls.matcher.match(cls.src_descriptors, cls.dst_descriptors)
        
        # Sort matches by distance
        cls.matches = sorted(cls.matches, key=lambda x: x.distance)
    
    def test_estimate_homography(self):
        """Test homography estimation."""
        H, mask, score = self.transformation.estimate_homography(
            self.src_keypoints,
            self.dst_keypoints,
            self.matches[:30]  # Use top 30 matches
        )
        
        self.assertIsNotNone(H)
        self.assertIsNotNone(mask)
        self.assertGreaterEqual(score, 0.5)
    
    def test_refine_homography(self):
        """Test homography refinement."""
        H, mask, score = self.transformation.estimate_homography(
            self.src_keypoints,
            self.dst_keypoints,
            self.matches[:30]
        )
        
        refined_H = self.transformation.refine_homography(
            H,
            self.src_keypoints,
            self.dst_keypoints,
            self.matches[:30],
            mask
        )
        
        self.assertIsNotNone(refined_H)
    
    def test_transform_points(self):
        """Test point transformation."""
        H, mask, score = self.transformation.estimate_homography(
            self.src_keypoints,
            self.dst_keypoints,
            self.matches[:30]
        )
        
        # Points to transform
        points = np.array([
            [50, 50],
            [350, 50],
            [350, 250],
            [50, 250]
        ])
        
        transformed = self.transformation.transform_points(points, H)
        
        self.assertEqual(transformed.shape, points.shape)
    
    def test_transform_roi(self):
        """Test ROI transformation."""
        H, mask, score = self.transformation.estimate_homography(
            self.src_keypoints,
            self.dst_keypoints,
            self.matches[:30]
        )
        
        # ROI to transform
        roi = np.array([50, 50, 350, 250])
        
        transformed_roi = self.transformation.transform_roi(roi, H)
        
        self.assertEqual(transformed_roi.shape, roi.shape)
    
    def test_warp_image(self):
        """Test image warping."""
        H, mask, score = self.transformation.estimate_homography(
            self.src_keypoints,
            self.dst_keypoints,
            self.matches[:30]
        )
        
        warped = self.transformation.warp_image(self.src_image, H, self.dst_image.shape)
        
        self.assertEqual(warped.shape, self.dst_image.shape)
    
    def test_is_homography_valid(self):
        """Test homography validation."""
        H, mask, score = self.transformation.estimate_homography(
            self.src_keypoints,
            self.dst_keypoints,
            self.matches[:30]
        )
        
        valid = self.transformation.is_homography_valid(H, self.src_image.shape, self.dst_image.shape)
        
        self.assertTrue(valid)
    
    def test_handle_perspective_distortion(self):
        """Test perspective distortion handling."""
        # Create image with perspective distortion
        src_points = np.array([
            [50, 50],
            [350, 60],
            [320, 250],
            [70, 240]
        ], dtype=np.float32)
        
        dst_points = np.array([
            [50, 50],
            [350, 50],
            [350, 250],
            [50, 250]
        ], dtype=np.float32)
        
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        distorted = cv2.warpPerspective(self.src_image, np.linalg.inv(H), self.src_image.shape[1::-1])
        
        corrected, correction_H = self.transformation.handle_perspective_distortion(distorted)
        
        self.assertIsNotNone(correction_H)


if __name__ == '__main__':
    unittest.main()
