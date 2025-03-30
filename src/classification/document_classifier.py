"""
Document classifier for distinguishing between pharmaceutical packages and information sheets.
Uses lightweight feature extraction for rapid classification.
"""

import os
import time
import numpy as np
import cv2
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from loguru import logger

from ..utils.performance_monitor import PerformanceTimer
from ..utils.image_preprocessor import normalize_image, resize_with_aspect_ratio
from .feature_extractor import FeatureExtractor

class DocumentClassifier:
    """
    Classifies pharmaceutical document images as either packages or information sheets.
    Uses a lightweight CNN classifier with HOG feature extraction for speed.
    """

    # Document types
    PACKAGE = 0
    INFORMATION_SHEET = 1
    
    def __init__(self, config_path='config/system_config.json', model_path='models/classifier/document_classifier'):
        """
        Initialize the document classifier.
        
        Args:
            config_path (str): Path to configuration file
            model_path (str): Path to pre-trained classification model
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.classifier_config = config['classification']
            self.hardware_config = config['hardware']
        
        self.confidence_threshold = self.classifier_config.get('confidence_threshold', 0.85)
        self.use_gpu = self.hardware_config.get('use_gpu', True)
        self.feature_type = self.classifier_config.get('feature_type', 'hog')
        self.use_lightweight_model = self.classifier_config.get('use_lightweight_model', True)
        self.fallback_to_template_matching = self.classifier_config.get('fallback_to_template_matching', True)
        
        # Feature extractor for image preprocessing
        self.feature_extractor = FeatureExtractor(feature_type=self.feature_type)
        
        # Initialize model (build or load)
        if os.path.exists(f"{model_path}.pdmodel"):
            self._load_model(model_path)
        else:
            self._build_model()
            # Save the model for future use
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            paddle.jit.save(self.model, model_path)
        
        # Model to inference mode
        self.model.eval()
        
        # Performance timer
        self.timer = PerformanceTimer()
        
        logger.info(f"Document classifier initialized with feature type: {self.feature_type}")
    
    def _build_model(self):
        """Build the classifier model with optimized architecture for ARM and Ampere."""
        # Define a lightweight model optimized for speed
        if self.use_lightweight_model:
            self.model = nn.Sequential(
                nn.Linear(self.feature_extractor.feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            # More complex model for higher accuracy
            self.model = nn.Sequential(
                nn.Linear(self.feature_extractor.feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        
        # Convert model to static graph for inference optimization
        self.model = paddle.jit.to_static(self.model)
    
    def _load_model(self, model_path):
        """
        Load a pre-trained classification model.
        
        Args:
            model_path (str): Path to the model
        """
        try:
            self.model = paddle.jit.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._build_model()
            logger.info("Falling back to building a new model")
    
    def preprocess_image(self, image):
        """
        Preprocess the image for classification.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image features
        """
        # Resize image for consistent processing
        resized = resize_with_aspect_ratio(image, max_size=512)
        
        # Convert to grayscale for feature extraction
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Document-specific preprocessing
        # Enhance contrast to better distinguish text density patterns
        gray = cv2.equalizeHist(gray)
        
        # Extract features using the feature extractor
        features = self.feature_extractor.extract(gray)
        
        # Convert to paddle tensor
        if self.use_gpu:
            return paddle.to_tensor(features, dtype='float32').reshape([1, -1])
        else:
            return paddle.to_tensor(features, dtype='float32').reshape([1, -1])
    
    def _analyze_text_density(self, image):
        """
        Analyze text density patterns which differ between packages and sheets.
        Used as a fallback classification method.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            float: Score indicating likelihood of being an information sheet (0-1)
        """
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for consistent processing
        gray = resize_with_aspect_ratio(gray, max_size=512)
        
        # Apply adaptive thresholding to highlight text
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find text contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to eliminate noise
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10 and cv2.contourArea(cnt) < 1000]
        
        if not valid_contours:
            return 0.5  # Neutral if no valid contours
        
        # Calculate metrics
        contour_count = len(valid_contours)
        
        # Information sheets typically have many small text contours
        # Packages typically have fewer, larger contours
        
        # Analyze contour density (number of contours per unit area)
        image_area = gray.shape[0] * gray.shape[1]
        contour_density = contour_count / (image_area / 1000)  # Normalized per 1000 pixels
        
        # Analyze contour size distribution
        areas = [cv2.contourArea(cnt) for cnt in valid_contours]
        if not areas:
            return 0.5
            
        mean_area = np.mean(areas)
        
        # Calculate text line metrics by clustering y-coordinates
        y_centers = [np.mean(cnt[:, 0, 1]) for cnt in valid_contours]
        y_sorted = np.sort(y_centers)
        
        # Count text lines by clustering y-coordinates
        line_threshold = gray.shape[0] * 0.02  # 2% of image height
        line_count = 1
        for i in range(1, len(y_sorted)):
            if y_sorted[i] - y_sorted[i-1] > line_threshold:
                line_count += 1
        
        # Calculate features
        line_density = line_count / (gray.shape[0] / 100)  # Lines per 100 pixels of height
        
        # Score calculation (higher = more likely to be information sheet)
        # Information sheets have higher contour density, smaller mean area, higher line density
        
        # Normalize and combine metrics
        contour_density_score = min(1.0, contour_density / 5.0)  # Normalize to 0-1 range
        mean_area_score = 1.0 - min(1.0, mean_area / 200.0)  # Smaller areas score higher
        line_density_score = min(1.0, line_density / 5.0)  # More lines score higher
        
        # Final score is weighted combination
        sheet_score = (0.4 * contour_density_score + 
                      0.3 * mean_area_score + 
                      0.3 * line_density_score)
        
        return sheet_score
    
    def _analyze_layout(self, image):
        """
        Analyze document layout which differs between packages and sheets.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            float: Score indicating likelihood of being an information sheet (0-1)
        """
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for consistent processing
        gray = resize_with_aspect_ratio(gray, max_size=512)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find straight lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        # Analyze line properties
        if lines is None:
            return 0.5  # Neutral
        
        # Count horizontal and vertical lines
        horizontal_count = 0
        vertical_count = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 20 or angle > 160:
                horizontal_count += 1
            elif 70 < angle < 110:
                vertical_count += 1
        
        # Information sheets typically have more horizontal lines for text
        # and structured vertical lines for columns/sections
        h_ratio = horizontal_count / max(1, len(lines))
        v_ratio = vertical_count / max(1, len(lines))
        
        # Calculate grid score - higher for structured grids (common in info sheets)
        grid_score = min(1.0, (h_ratio * v_ratio * 4))
        
        # Analyze text margins
        # Find left margin by analyzing left edges of text blocks
        left_positions = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if 70 < np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) < 110:
                left_positions.append(min(x1, x2))
        
        if not left_positions:
            margin_alignment = 0.5
        else:
            # Calculate alignment of left margins
            left_positions = np.array(left_positions)
            left_mean = np.mean(left_positions)
            left_std = np.std(left_positions)
            
            # Normalized std deviation (lower std = better alignment = higher score)
            normalized_std = min(left_std / (gray.shape[1] * 0.1), 1.0)
            margin_alignment = 1.0 - normalized_std
        
        # Final score (higher = more likely to be information sheet)
        return 0.4 * grid_score + 0.6 * margin_alignment
    
    def classify(self, image):
        """
        Classify an image as either a package or information sheet.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (document_type, confidence)
                document_type: PACKAGE or INFORMATION_SHEET
                confidence: Classification confidence (0-1)
        """
        self.timer.start("document_classification")
        
        # Preprocess image and extract features
        features = self.preprocess_image(image)
        
        # Run inference
        with paddle.no_grad():
            logits = self.model(features)
            probabilities = F.softmax(logits, axis=1)
            
            # Get prediction and confidence
            pred_class = int(paddle.argmax(probabilities, axis=1).numpy()[0])
            confidence = float(probabilities.numpy()[0][pred_class])
        
        # If confidence is low, use fallback methods
        if confidence < self.confidence_threshold:
            logger.debug(f"Low classification confidence ({confidence:.3f}), using fallback methods")
            
            # Fallback to text density analysis
            text_density_score = self._analyze_text_density(image)
            layout_score = self._analyze_layout(image)
            
            # Combine scores (weighted average)
            fallback_sheet_score = 0.6 * text_density_score + 0.4 * layout_score
            
            # Threshold to determine class
            if fallback_sheet_score > 0.55:
                pred_class = self.INFORMATION_SHEET
                confidence = fallback_sheet_score
            else:
                pred_class = self.PACKAGE
                confidence = 1.0 - fallback_sheet_score
            
            logger.debug(f"Fallback classification: {'Sheet' if pred_class == self.INFORMATION_SHEET else 'Package'} "
                       f"with confidence {confidence:.3f}")
        
        elapsed_ms = self.timer.stop("document_classification")
        logger.debug(f"Document classification completed in {elapsed_ms:.2f}ms with confidence {confidence:.3f}")
        
        return pred_class, confidence
    
    def classify_batch(self, images):
        """
        Classify a batch of images.
        
        Args:
            images (list): List of input images
            
        Returns:
            list: List of (document_type, confidence) tuples
        """
        self.timer.start("batch_classification")
        
        results = []
        for image in images:
            results.append(self.classify(image))
        
        elapsed_ms = self.timer.stop("batch_classification")
        logger.debug(f"Batch classification of {len(images)} images completed in {elapsed_ms:.2f}ms")
        
        return results
    
    def is_package(self, image, min_confidence=0.7):
        """
        Determine if an image is a package.
        
        Args:
            image (numpy.ndarray): Input image
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            bool: True if image is a package, False otherwise
        """
        doc_type, confidence = self.classify(image)
        return doc_type == self.PACKAGE and confidence >= min_confidence
    
    def is_information_sheet(self, image, min_confidence=0.7):
        """
        Determine if an image is an information sheet.
        
        Args:
            image (numpy.ndarray): Input image
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            bool: True if image is an information sheet, False otherwise
        """
        doc_type, confidence = self.classify(image)
        return doc_type == self.INFORMATION_SHEET and confidence >= min_confidence
