"""
Feature extractor for document classification and template matching.
"""

import cv2
import numpy as np
from loguru import logger
from skimage.feature import hog
from skimage import exposure
import paddle

class FeatureExtractor:
    """
    Extracts image features for document classification and template matching.
    Supports multiple feature types with optimizations for pharmaceutical packages and sheets.
    """
    
    def __init__(self, feature_type='hog', max_features=500):
        """
        Initialize the feature extractor.
        
        Args:
            feature_type (str): Feature extraction method ('hog', 'orb', 'sift', 'custom')
            max_features (int): Maximum number of features to extract (for keypoint-based methods)
        """
        self.feature_type = feature_type.lower()
        self.max_features = max_features
        
        # Initialize feature detector based on type
        if self.feature_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=self.max_features,
                                          scaleFactor=1.2,
                                          nlevels=8,
                                          edgeThreshold=31,
                                          firstLevel=0,
                                          WTA_K=2,
                                          scoreType=cv2.ORB_HARRIS_SCORE,
                                          patchSize=31,
                                          fastThreshold=20)
            self.feature_dim = 32  # ORB descriptor size
        elif self.feature_type == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=self.max_features,
                                           sigma=1.6,
                                           nOctaveLayers=3)
            self.feature_dim = 128  # SIFT descriptor size
        elif self.feature_type == 'hog':
            # HOG parameters optimized for document classification
            self.orientations = 9
            self.pixels_per_cell = (8, 8)
            self.cells_per_block = (3, 3)
            self.block_norm = 'L2-Hys'
            self.feature_dim = 5184  # 9 orientations * 24*24 cells (for 256x256 image)
        elif self.feature_type == 'custom':
            # Custom hybrid feature extractor
            self.orb_detector = cv2.ORB_create(nfeatures=self.max_features // 2)
            self.feature_dim = 5184 + 32*50  # HOG features + 50 ORB descriptors
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        logger.info(f"Feature extractor initialized with {feature_type} method")
    
    def extract(self, image):
        """
        Extract features from an image.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Extracted features
        """
        if image is None:
            logger.error("Cannot extract features from None image")
            return np.zeros((1, self.feature_dim), dtype=np.float32)
        
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Make sure image has proper dimensions
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            image = cv2.resize(image, (512, 512))
        
        # Extract appropriate features based on the selected method
        if self.feature_type == 'orb':
            return self._extract_orb(image)
        elif self.feature_type == 'sift':
            return self._extract_sift(image)
        elif self.feature_type == 'hog':
            return self._extract_hog(image)
        elif self.feature_type == 'custom':
            return self._extract_custom(image)
    
    def _extract_orb(self, image):
        """
        Extract ORB features.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: ORB descriptors
        """
        # Find keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        if descriptors is None or len(keypoints) == 0:
            # If no features detected, return zeros
            return np.zeros((1, self.feature_dim), dtype=np.float32)
        
        # Create a fixed-size feature vector
        # For simplicity, we'll use the mean of all descriptors
        # In a more sophisticated approach, we might use bag-of-words or other methods
        mean_descriptor = np.mean(descriptors, axis=0).astype(np.float32)
        
        return mean_descriptor.reshape(1, -1)
    
    def _extract_sift(self, image):
        """
        Extract SIFT features.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: SIFT descriptors
        """
        # Find keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        if descriptors is None or len(keypoints) == 0:
            # If no features detected, return zeros
            return np.zeros((1, self.feature_dim), dtype=np.float32)
        
        # Create a fixed-size feature vector (mean of descriptors)
        mean_descriptor = np.mean(descriptors, axis=0).astype(np.float32)
        
        return mean_descriptor.reshape(1, -1)
    
    def _extract_hog(self, image):
        """
        Extract HOG features optimized for document classification.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: HOG features
        """
        # Resize to standard size for consistent feature dimensions
        image = cv2.resize(image, (256, 256))
        
        # Calculate HOG features
        features = hog(image, 
                      orientations=self.orientations,
                      pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block,
                      block_norm=self.block_norm,
                      visualize=False,
                      transform_sqrt=True)
        
        return features.reshape(1, -1).astype(np.float32)
    
    def _extract_custom(self, image):
        """
        Extract custom hybrid features combining HOG and ORB.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Combined feature vector
        """
        # Get HOG features
        hog_features = self._extract_hog(image).flatten()
        
        # Get ORB keypoints and descriptors
        keypoints, descriptors = self.orb_detector.detectAndCompute(image, None)
        
        # If no ORB features detected, use zeros
        if descriptors is None or len(descriptors) == 0:
            orb_features = np.zeros((50, 32), dtype=np.float32)
        else:
            # Take up to 50 descriptors, pad if fewer
            if len(descriptors) >= 50:
                orb_features = descriptors[:50]
            else:
                # Pad with zeros
                orb_features = np.zeros((50, 32), dtype=np.float32)
                orb_features[:len(descriptors)] = descriptors
        
        # Flatten ORB features
        orb_features = orb_features.flatten()
        
        # Combine features
        combined = np.concatenate([hog_features, orb_features])
        
        return combined.reshape(1, -1).astype(np.float32)
    
    def extract_keypoints(self, image):
        """
        Extract keypoints and descriptors for template matching.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        if image is None:
            logger.error("Cannot extract keypoints from None image")
            return [], None
        
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract keypoints and descriptors based on the selected method
        if self.feature_type == 'orb' or self.feature_type == 'custom':
            detector = self.detector if self.feature_type == 'orb' else self.orb_detector
            keypoints, descriptors = detector.detectAndCompute(image, None)
        elif self.feature_type == 'sift':
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
        else:
            # For HOG, we need to simulate keypoints for compatibility
            # This is not optimal for template matching but allows consistent API
            keypoints = [cv2.KeyPoint(x, y, 10) for y in range(0, image.shape[0], 10) 
                                               for x in range(0, image.shape[1], 10)]
            descriptors = self._extract_hog(image)
        
        return keypoints, descriptors
    
    def compute_descriptors(self, image, keypoints):
        """
        Compute descriptors for given keypoints.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            keypoints (list): List of keypoints
            
        Returns:
            numpy.ndarray: Descriptors
        """
        if image is None or not keypoints:
            logger.error("Cannot compute descriptors from None image or empty keypoints")
            return None
        
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute descriptors based on the selected method
        if self.feature_type == 'orb' or self.feature_type == 'custom':
            detector = self.detector if self.feature_type == 'orb' else self.orb_detector
            _, descriptors = detector.compute(image, keypoints)
        elif self.feature_type == 'sift':
            _, descriptors = self.detector.compute(image, keypoints)
        else:
            # For HOG, we don't support this operation properly
            descriptors = None
        
        return descriptors
    
    def to_paddle_tensor(self, features):
        """
        Convert features to PaddlePaddle tensor.
        
        Args:
            features (numpy.ndarray): Feature array
            
        Returns:
            paddle.Tensor: PaddlePaddle tensor
        """
        return paddle.to_tensor(features, dtype='float32')
