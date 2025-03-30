"""
Feature-based template matcher for pharmaceutical package and sheet recognition.
"""

import os
import cv2
import numpy as np
import paddle
import time
from loguru import logger
import joblib
from concurrent.futures import ThreadPoolExecutor

from ..utils.performance_monitor import PerformanceTimer
from ..utils.image_preprocessor import normalize_image, resize_with_aspect_ratio
from ..classification.feature_extractor import FeatureExtractor
from .template_indexer import TemplateIndexer

class FeatureMatcher:
    """
    Matches input images to templates using feature-based matching.
    Optimized for pharmaceutical packages and information sheets.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize the feature matcher.
        
        Args:
            config_path (str): Path to the system configuration file
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.template_config = config['template_matching']
            self.hardware_config = config['hardware']
            self.paths = config['paths']
        
        # Configure feature extraction
        self.feature_type = self.template_config.get('feature_extractor', 'orb')
        self.max_features = self.template_config.get('max_features', 500)
        self.feature_extractor = FeatureExtractor(feature_type=self.feature_type, 
                                                 max_features=self.max_features)
        
        # Configure matching parameters
        self.matching_algorithm = self.template_config.get('matching_algorithm', 'flann')
        self.ransac_threshold = self.template_config.get('ransac_threshold', 5.0)
        self.min_match_count = self.template_config.get('min_match_count', 10)
        self.batch_size = self.template_config.get('batch_size', 8)
        
        # Load or create template indexers
        self.package_indexer = TemplateIndexer(self.feature_extractor, 
                                              os.path.join(self.paths['templates_dir'], 'packages'),
                                              self.template_config.get('index_type', 'lmdb'))
        
        self.sheet_indexer = TemplateIndexer(self.feature_extractor,
                                            os.path.join(self.paths['templates_dir'], 'sheets'),
                                            self.template_config.get('index_type', 'lmdb'))
        
        # Configure GPU usage
        self.use_gpu = self.hardware_config.get('use_gpu', True)
        
        # Initialize matchers
        self._init_matchers()
        
        # Performance timer
        self.timer = PerformanceTimer()
        
        # Thread pool for parallel matching
        self.use_parallel = self.hardware_config.get('cpu_cores', 6) > 1
        if self.use_parallel:
            self.executor = ThreadPoolExecutor(max_workers=min(4, self.hardware_config.get('cpu_cores', 6)))
        
        logger.info(f"Feature matcher initialized with {self.feature_type} features and "
                   f"{self.matching_algorithm} matching algorithm")
    
    def _init_matchers(self):
        """Initialize feature matchers based on the selected algorithm."""
        if self.feature_type == 'orb':
            # For binary descriptors like ORB
            if self.matching_algorithm == 'flann':
                # FLANN parameters for ORB
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                   table_number=12,
                                   key_size=20,
                                   multi_probe_level=2)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                # Brute force matcher with Hamming distance for binary descriptors
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # For float descriptors like SIFT or HOG
            if self.matching_algorithm == 'flann':
                # FLANN parameters for SIFT/float descriptors
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                # Brute force matcher with L2 distance for float descriptors
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def preprocess_image(self, image):
        """
        Preprocess image for feature matching.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (processed_image, keypoints, descriptors)
        """
        # Resize if necessary while maintaining aspect ratio
        if max(image.shape[0], image.shape[1]) > 1000:
            image = resize_with_aspect_ratio(image, max_size=1000)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply contrast enhancement for better feature detection
        gray = cv2.equalizeHist(gray)
        
        # Extract keypoints and descriptors
        keypoints, descriptors = self.feature_extractor.extract_keypoints(gray)
        
        return gray, keypoints, descriptors
    
    def _match_descriptors(self, desc1, desc2):
        """
        Match descriptors using the configured matcher.
        
        Args:
            desc1 (numpy.ndarray): Query descriptors
            desc2 (numpy.ndarray): Train descriptors
            
        Returns:
            list: List of matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []
        
        # Ensure correct type (some matchers require specific types)
        if self.feature_type == 'orb':
            if desc1.dtype != np.uint8:
                desc1 = np.uint8(desc1)
            if desc2.dtype != np.uint8:
                desc2 = np.uint8(desc2)
        
        # Match descriptors
        if self.matching_algorithm == 'flann':
            # FLANN matcher can sometimes crash with certain inputs
            try:
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
                # Filter matches using Lowe's ratio test
                good_matches = []
                for match_group in matches:
                    if len(match_group) >= 2:
                        m, n = match_group
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                return good_matches
            except Exception as e:
                logger.warning(f"FLANN matching failed: {e}. Falling back to BFMatcher.")
                # Fallback to brute force if FLANN fails
                bf = cv2.BFMatcher(cv2.NORM_HAMMING if self.feature_type == 'orb' else cv2.NORM_L2)
                matches = bf.match(desc1, desc2)
                return sorted(matches, key=lambda x: x.distance)[:30]
        else:
            # Brute force matching
            matches = self.matcher.match(desc1, desc2)
            return sorted(matches, key=lambda x: x.distance)[:30]
    
    def _find_homography(self, keypoints1, keypoints2, matches):
        """
        Find homography matrix between two images.
        
        Args:
            keypoints1 (list): Query keypoints
            keypoints2 (list): Train keypoints
            matches (list): List of matches
            
        Returns:
            tuple: (homography_matrix, inlier_mask, match_score)
        """
        if len(matches) < self.min_match_count:
            return None, None, 0.0
        
        # Extract points from keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        # Calculate match score
        inliers = np.sum(mask) if mask is not None else 0
        match_score = inliers / len(matches) if matches else 0.0
        
        return H, mask, match_score
    
    def match_template(self, query_image, template_id=None, is_sheet=False):
        """
        Match an image against stored templates.
        
        Args:
            query_image (numpy.ndarray): Query image
            template_id (str, optional): Specific template ID to match against
            is_sheet (bool): True if matching against sheet templates, False for packages
            
        Returns:
            tuple: (best_match_id, confidence, homography_matrix)
        """
        self.timer.start("template_matching")
        
        # Select the appropriate indexer
        indexer = self.sheet_indexer if is_sheet else self.package_indexer
        
        # Preprocess query image
        _, query_keypoints, query_descriptors = self.preprocess_image(query_image)
        
        if query_descriptors is None or len(query_keypoints) < 10:
            logger.warning("Not enough features found in query image")
            return None, 0.0, None
        
        best_match_id = None
        best_confidence = 0.0
        best_homography = None
        
        # If template_id is specified, only match against that template
        if template_id:
            template_info = indexer.get_template(template_id)
            if template_info:
                template_keypoints = template_info['keypoints']
                template_descriptors = template_info['descriptors']
                
                # Match descriptors
                matches = self._match_descriptors(query_descriptors, template_descriptors)
                
                if len(matches) >= self.min_match_count:
                    # Find homography
                    H, mask, score = self._find_homography(query_keypoints, template_keypoints, matches)
                    
                    if H is not None:
                        best_match_id = template_id
                        best_confidence = score
                        best_homography = H
            
        else:
            # Match against all templates
            if self.use_parallel:
                results = self._match_parallel(query_keypoints, query_descriptors, indexer)
            else:
                results = self._match_sequential(query_keypoints, query_descriptors, indexer)
            
            # Find best match
            if results:
                best_match = max(results, key=lambda x: x[1])
                best_match_id, best_confidence, best_homography = best_match
        
        elapsed_ms = self.timer.stop("template_matching")
        logger.debug(f"Template matching completed in {elapsed_ms:.2f}ms, "
                   f"best match: {best_match_id}, confidence: {best_confidence:.3f}")
        
        return best_match_id, best_confidence, best_homography
    
    def _match_sequential(self, query_keypoints, query_descriptors, indexer):
        """
        Match query against all templates sequentially.
        
        Args:
            query_keypoints (list): Query keypoints
            query_descriptors (numpy.ndarray): Query descriptors
            indexer (TemplateIndexer): Template indexer
            
        Returns:
            list: List of (template_id, confidence, homography) tuples
        """
        results = []
        
        for template_id in indexer.get_all_template_ids():
            template_info = indexer.get_template(template_id)
            if not template_info:
                continue
            
            template_keypoints = template_info['keypoints']
            template_descriptors = template_info['descriptors']
            
            # Match descriptors
            matches = self._match_descriptors(query_descriptors, template_descriptors)
            
            if len(matches) >= self.min_match_count:
                # Find homography
                H, mask, score = self._find_homography(query_keypoints, template_keypoints, matches)
                
                if H is not None:
                    results.append((template_id, score, H))
        
        return results
    
    def _match_parallel(self, query_keypoints, query_descriptors, indexer):
        """
        Match query against all templates in parallel.
        
        Args:
            query_keypoints (list): Query keypoints
            query_descriptors (numpy.ndarray): Query descriptors
            indexer (TemplateIndexer): Template indexer
            
        Returns:
            list: List of (template_id, confidence, homography) tuples
        """
        template_ids = indexer.get_all_template_ids()
        
        # Create batches
        batches = [template_ids[i:i+self.batch_size] for i in range(0, len(template_ids), self.batch_size)]
        
        all_results = []
        
        for batch in batches:
            # Process each batch in parallel
            futures = []
            for template_id in batch:
                future = self.executor.submit(self._match_single_template, 
                                             query_keypoints, 
                                             query_descriptors,
                                             template_id,
                                             indexer)
                futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                if result:
                    all_results.append(result)
        
        return all_results
    
    def _match_single_template(self, query_keypoints, query_descriptors, template_id, indexer):
        """
        Match query against a single template.
        
        Args:
            query_keypoints (list): Query keypoints
            query_descriptors (numpy.ndarray): Query descriptors
            template_id (str): Template ID
            indexer (TemplateIndexer): Template indexer
            
        Returns:
            tuple: (template_id, confidence, homography) or None if no match
        """
        template_info = indexer.get_template(template_id)
        if not template_info:
            return None
        
        template_keypoints = template_info['keypoints']
        template_descriptors = template_info['descriptors']
        
        # Match descriptors
        matches = self._match_descriptors(query_descriptors, template_descriptors)
        
        if len(matches) >= self.min_match_count:
            # Find homography
            H, mask, score = self._find_homography(query_keypoints, template_keypoints, matches)
            
            if H is not None:
                return (template_id, score, H)
        
        return None
    
    def identify_template(self, image, is_sheet=None):
        """
        Identify the template for an input image.
        If is_sheet is None, will try both package and sheet templates.
        
        Args:
            image (numpy.ndarray): Input image
            is_sheet (bool, optional): True for sheet, False for package, None for auto
            
        Returns:
            tuple: (template_id, template_type, confidence, homography)
                template_type: 'package' or 'sheet'
        """
        if is_sheet is None:
            # Try both types
            package_result = self.match_template(image, is_sheet=False)
            sheet_result = self.match_template(image, is_sheet=True)
            
            package_id, package_confidence, package_homography = package_result
            sheet_id, sheet_confidence, sheet_homography = sheet_result
            
            # Compare confidences
            if package_confidence > sheet_confidence:
                return package_id, 'package', package_confidence, package_homography
            else:
                return sheet_id, 'sheet', sheet_confidence, sheet_homography
        else:
            # Match against specific type
            template_id, confidence, homography = self.match_template(image, is_sheet=is_sheet)
            template_type = 'sheet' if is_sheet else 'package'
            return template_id, template_type, confidence, homography
    
    def add_template(self, image, template_id, is_sheet=False):
        """
        Add a new template to the database.
        
        Args:
            image (numpy.ndarray): Template image
            template_id (str): Unique template ID
            is_sheet (bool): True for sheet, False for package
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        # Preprocess template
        _, keypoints, descriptors = self.preprocess_image(image)
        
        if descriptors is None or len(keypoints) < 10:
            logger.warning(f"Not enough features found in template {template_id}")
            return False
        
        # Select the appropriate indexer
        indexer = self.sheet_indexer if is_sheet else self.package_indexer
        
        # Add template to index
        return indexer.add_template(template_id, keypoints, descriptors, image)
    
    def remove_template(self, template_id, is_sheet=False):
        """
        Remove a template from the database.
        
        Args:
            template_id (str): Unique template ID
            is_sheet (bool): True for sheet, False for package
            
        Returns:
            bool: True if successfully removed, False otherwise
        """
        # Select the appropriate indexer
        indexer = self.sheet_indexer if is_sheet else self.package_indexer
        
        # Remove template from index
        return indexer.remove_template(template_id)
