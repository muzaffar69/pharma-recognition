"""
Transformation module for template-to-image mapping in pharmaceutical recognition.
"""

import cv2
import numpy as np
from loguru import logger
import paddle

from ..utils.performance_monitor import PerformanceTimer
from ..utils.cuda_utils import CUDAUtils

class TransformationEstimator:
    """
    Estimates transformation between template and input images.
    Handles perspective transformations, homographies, and ROI mapping.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize transformation estimator.
        
        Args:
            config_path (str): Path to configuration file
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.template_config = config['template_matching']
            self.hardware_config = config['hardware']
        
        # Configure transformation parameters
        self.ransac_threshold = self.template_config.get('ransac_threshold', 5.0)
        self.min_match_count = self.template_config.get('min_match_count', 10)
        
        # RANSAC parameters
        self.ransac_reproj_threshold = 5.0
        self.ransac_max_iter = 2000
        self.ransac_confidence = 0.995
        
        # GPU acceleration
        self.use_gpu = self.hardware_config.get('use_gpu', True)
        self.cuda_utils = CUDAUtils(config_path)
        
        # Performance timer
        self.timer = PerformanceTimer()
        
        logger.info("Transformation estimator initialized")
    
    def estimate_homography(self, keypoints1, keypoints2, matches):
        """
        Estimate homography matrix between two images.
        
        Args:
            keypoints1 (list): Keypoints from first image
            keypoints2 (list): Keypoints from second image
            matches (list): List of DMatch objects
            
        Returns:
            tuple: (homography_matrix, inlier_mask, match_score)
        """
        self.timer.start("homography_estimation")
        
        if len(matches) < self.min_match_count:
            logger.debug(f"Not enough matches to estimate homography: {len(matches)} < {self.min_match_count}")
            self.timer.stop("homography_estimation")
            return None, None, 0.0
        
        # Extract matched points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate homography using RANSAC
        try:
            H, mask = cv2.findHomography(
                src_pts, 
                dst_pts, 
                cv2.RANSAC, 
                self.ransac_reproj_threshold, 
                maxIters=self.ransac_max_iter, 
                confidence=self.ransac_confidence
            )
            
            if H is None:
                logger.debug("Failed to estimate homography")
                self.timer.stop("homography_estimation")
                return None, None, 0.0
            
            # Calculate match score (ratio of inliers to total matches)
            inliers = mask.ravel().sum()
            match_score = inliers / len(matches)
            
            elapsed_ms = self.timer.stop("homography_estimation")
            logger.debug(f"Homography estimated in {elapsed_ms:.2f}ms with score {match_score:.3f} ({inliers}/{len(matches)} inliers)")
            
            return H, mask, match_score
            
        except Exception as e:
            logger.error(f"Error estimating homography: {e}")
            self.timer.stop("homography_estimation")
            return None, None, 0.0
    
    def refine_homography(self, H, keypoints1, keypoints2, matches, mask):
        """
        Refine homography using only inliers.
        
        Args:
            H (numpy.ndarray): Initial homography matrix
            keypoints1 (list): Keypoints from first image
            keypoints2 (list): Keypoints from second image
            matches (list): List of DMatch objects
            mask (numpy.ndarray): Inlier mask
            
        Returns:
            numpy.ndarray: Refined homography matrix
        """
        if H is None or mask is None:
            return H
        
        # Extract inlier matches
        inlier_matches = []
        for i, m in enumerate(matches):
            if mask[i][0]:
                inlier_matches.append(m)
        
        if len(inlier_matches) < 4:
            logger.debug(f"Not enough inliers for homography refinement: {len(inlier_matches)}")
            return H
        
        # Extract inlier points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        
        # Calculate refined homography
        try:
            refined_H, _ = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
            
            if refined_H is None:
                logger.debug("Failed to refine homography")
                return H
            
            return refined_H
            
        except Exception as e:
            logger.error(f"Error refining homography: {e}")
            return H
    
    def transform_points(self, points, homography):
        """
        Transform points using homography matrix.
        
        Args:
            points (numpy.ndarray): Points to transform [n x 2]
            homography (numpy.ndarray): Homography matrix
            
        Returns:
            numpy.ndarray: Transformed points
        """
        if homography is None:
            return points
        
        # Convert to homogeneous coordinates
        n = len(points)
        points_homog = np.hstack((points, np.ones((n, 1)))).T
        
        # Apply homography
        transformed_homog = np.dot(homography, points_homog)
        
        # Convert back to Cartesian coordinates
        transformed = transformed_homog[:2] / transformed_homog[2]
        
        return transformed.T
    
    def transform_roi(self, roi, homography, target_shape=None):
        """
        Transform ROI using homography matrix.
        
        Args:
            roi (numpy.ndarray): ROI coordinates [x1, y1, x2, y2]
            homography (numpy.ndarray): Homography matrix
            target_shape (tuple, optional): Shape of target image (height, width)
            
        Returns:
            numpy.ndarray: Transformed ROI
        """
        if homography is None:
            return roi
        
        # Convert ROI to corner points
        x1, y1, x2, y2 = roi
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        
        # Transform corners
        transformed_corners = self.transform_points(corners, homography)
        
        # Get bounding box of transformed corners
        x_min = np.min(transformed_corners[:, 0])
        y_min = np.min(transformed_corners[:, 1])
        x_max = np.max(transformed_corners[:, 0])
        y_max = np.max(transformed_corners[:, 1])
        
        # Apply bounds checking if target shape is provided
        if target_shape is not None:
            height, width = target_shape
            x_min = max(0, min(width - 1, x_min))
            y_min = max(0, min(height - 1, y_min))
            x_max = max(0, min(width - 1, x_max))
            y_max = max(0, min(height - 1, y_max))
        
        return np.array([x_min, y_min, x_max, y_max])
    
    def warp_image(self, image, homography, target_shape):
        """
        Warp image using homography matrix.
        
        Args:
            image (numpy.ndarray): Source image
            homography (numpy.ndarray): Homography matrix
            target_shape (tuple): Shape of target image (height, width)
            
        Returns:
            numpy.ndarray: Warped image
        """
        if homography is None:
            return image
        
        # Get target dimensions
        if len(target_shape) == 3:
            height, width, channels = target_shape
        else:
            height, width = target_shape
        
        # Apply warp perspective
        try:
            warped = cv2.warpPerspective(image, homography, (width, height))
            return warped
        except Exception as e:
            logger.error(f"Error warping image: {e}")
            return image
    
    def is_homography_valid(self, homography, src_shape, dst_shape, max_scale=5.0):
        """
        Check if homography is valid (no extreme distortions).
        
        Args:
            homography (numpy.ndarray): Homography matrix
            src_shape (tuple): Shape of source image (height, width)
            dst_shape (tuple): Shape of destination image (height, width)
            max_scale (float): Maximum acceptable scale change
            
        Returns:
            bool: True if homography is valid
        """
        if homography is None:
            return False
        
        # Check for NaN or Inf values
        if np.any(np.isnan(homography)) or np.any(np.isinf(homography)):
            logger.debug("Homography contains NaN or Inf values")
            return False
        
        # Check determinant (should not be zero)
        det = np.linalg.det(homography[:2, :2])
        if abs(det) < 1e-6:
            logger.debug(f"Homography determinant too small: {det}")
            return False
        
        # Check if scale change is reasonable
        src_h, src_w = src_shape[:2]
        dst_h, dst_w = dst_shape[:2]
        
        # Create corner points for source image
        src_corners = np.array([
            [0, 0],
            [src_w, 0],
            [src_w, src_h],
            [0, src_h]
        ])
        
        # Transform corners to destination image
        dst_corners = self.transform_points(src_corners, homography)
        
        # Calculate areas
        src_area = src_w * src_h
        dst_area = cv2.contourArea(dst_corners.astype(np.float32))
        
        # Check scale change
        scale_change = abs(dst_area / src_area)
        if scale_change > max_scale or scale_change < 1/max_scale:
            logger.debug(f"Homography scale change too large: {scale_change}")
            return False
        
        return True
    
    def handle_perspective_distortion(self, image, homography=None):
        """
        Handle perspective distortion in image.
        
        Args:
            image (numpy.ndarray): Input image
            homography (numpy.ndarray, optional): Pre-computed homography
            
        Returns:
            tuple: (Corrected image, correction homography)
        """
        if homography is not None:
            # Already have homography
            return image, homography
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image, None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get approximate polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have a quadrilateral, correct perspective
        if len(approx) == 4:
            # Sort points by x+y value (top-left, top-right, bottom-right, bottom-left)
            points = approx.reshape(4, 2).astype(np.float32)
            points = points[np.argsort(np.sum(points, axis=1))]
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Define destination points (rectangle)
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Get perspective transform matrix
            H = cv2.getPerspectiveTransform(points, dst_points)
            
            # Apply perspective transformation
            corrected = cv2.warpPerspective(image, H, (width, height))
            
            return corrected, H
        
        return image, None
    
    def apply_gpu_optimization(self):
        """
        Apply GPU optimizations for transformation operations.
        """
        if not self.use_gpu:
            return
        
        # Initialize CUDA utilities for GPU optimization
        self.cuda_utils.optimize_for_inference()
        
        # Set optimal parameters for GPU
        self.ransac_max_iter = 3000  # More iterations possible on GPU
