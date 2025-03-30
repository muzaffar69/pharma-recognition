"""
Template matching configuration for pharmaceutical package and sheet recognition.
"""

import json
import os
from loguru import logger

class TemplateMatchingConfig:
    """
    Manages configuration settings for template matching operations.
    Optimizes feature extraction and matching for pharmaceutical packages and sheets.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize template matching configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.template_config = config['template_matching']
            self.hardware_config = config['hardware']
            self.paths = config['paths']
        
        # Feature extraction configuration
        self.feature_extractor = self.template_config.get('feature_extractor', 'orb')
        self.max_features = self.template_config.get('max_features', 500)
        
        # Matching configuration
        self.matching_algorithm = self.template_config.get('matching_algorithm', 'flann')
        self.ransac_threshold = self.template_config.get('ransac_threshold', 5.0)
        self.min_match_count = self.template_config.get('min_match_count', 10)
        self.index_type = self.template_config.get('index_type', 'lmdb')
        self.batch_size = self.template_config.get('batch_size', 8)
        
        # GPU acceleration
        self.use_gpu = self.hardware_config.get('use_gpu', True)
        self.use_tensor_cores = self.hardware_config.get('tensor_cores_enabled', True)
        
        # Template storage paths
        self.templates_dir = os.path.join(self.paths['templates_dir'])
        self.package_templates_dir = os.path.join(self.templates_dir, 'packages')
        self.sheet_templates_dir = os.path.join(self.templates_dir, 'sheets')
        
        logger.info("Template matching configuration initialized")
    
    def get_feature_params(self, feature_type=None):
        """
        Get parameters for feature extraction.
        
        Args:
            feature_type (str, optional): Override feature type
            
        Returns:
            dict: Feature extraction parameters
        """
        feat_type = feature_type or self.feature_extractor
        
        if feat_type == 'orb':
            return {
                'nfeatures': self.max_features,
                'scaleFactor': 1.2,
                'nlevels': 8,
                'edgeThreshold': 31,
                'firstLevel': 0,
                'WTA_K': 2,
                'scoreType': 'HARRIS_SCORE',
                'patchSize': 31,
                'fastThreshold': 20
            }
        elif feat_type == 'sift':
            return {
                'nfeatures': self.max_features,
                'sigma': 1.6,
                'nOctaveLayers': 3,
                'contrastThreshold': 0.04,
                'edgeThreshold': 10,
                'sigma': 1.6
            }
        elif feat_type == 'hog':
            return {
                'orientations': 9,
                'pixels_per_cell': (8, 8),
                'cells_per_block': (3, 3),
                'block_norm': 'L2-Hys'
            }
        else:
            logger.warning(f"Unknown feature type: {feat_type}, using default ORB")
            return self.get_feature_params('orb')
    
    def get_matcher_params(self, matcher_type=None):
        """
        Get parameters for feature matching.
        
        Args:
            matcher_type (str, optional): Override matcher type
            
        Returns:
            dict: Matcher parameters
        """
        match_type = matcher_type or self.matching_algorithm
        
        if match_type == 'flann':
            if self.feature_extractor == 'orb':
                # FLANN parameters for binary descriptors
                return {
                    'algorithm': 'LSH',
                    'table_number': 12,
                    'key_size': 20,
                    'multi_probe_level': 2,
                    'checks': 50
                }
            else:
                # FLANN parameters for float descriptors
                return {
                    'algorithm': 'KDTREE',
                    'trees': 5,
                    'checks': 50
                }
        elif match_type == 'bf':
            # Brute force matcher
            return {
                'crossCheck': True,
                'norm_type': 'HAMMING' if self.feature_extractor == 'orb' else 'L2'
            }
        else:
            logger.warning(f"Unknown matcher type: {match_type}, using default FLANN")
            return self.get_matcher_params('flann')
    
    def get_transformation_params(self):
        """
        Get parameters for template-to-image transformation.
        
        Returns:
            dict: Transformation parameters
        """
        return {
            'ransac_threshold': self.ransac_threshold,
            'min_match_count': self.min_match_count,
            'ransac_reproj_threshold': 5.0,
            'ransac_max_iter': 2000,
            'ransac_confidence': 0.995,
            'use_gpu': self.use_gpu
        }
    
    def get_package_specific_params(self):
        """
        Get package-specific template matching parameters.
        
        Returns:
            dict: Package-specific parameters
        """
        return {
            'min_match_count': max(8, self.min_match_count - 2),  # Lower threshold for packages
            'ransac_threshold': self.ransac_threshold * 1.2,  # More permissive threshold
            'handle_reflective_surfaces': True,
            'handle_perspective_distortion': True,
            'detect_curved_text': True
        }
    
    def get_sheet_specific_params(self):
        """
        Get sheet-specific template matching parameters.
        
        Returns:
            dict: Sheet-specific parameters
        """
        return {
            'min_match_count': self.min_match_count,  # Standard threshold
            'ransac_threshold': self.ransac_threshold,
            'detect_paragraphs': True,
            'detect_sections': True,
            'handle_dense_text': True
        }
    
    def optimize_for_hardware(self):
        """
        Optimize configuration for current hardware.
        
        Returns:
            dict: Hardware-specific optimizations
        """
        # ARM-specific optimizations
        arm_optimizations = {}
        if self.hardware_config.get('use_arm_optimization', True):
            arm_optimizations = {
                'thread_affinity': True,
                'simd_enabled': True,
                'optimize_memory_layout': True
            }
        
        # GPU-specific optimizations
        gpu_optimizations = {}
        if self.use_gpu:
            gpu_optimizations = {
                'use_gpu': True,
                'use_tensor_cores': self.use_tensor_cores,
                'use_mixed_precision': True,
                'optimized_memory_transfer': True
            }
        
        return {**arm_optimizations, **gpu_optimizations}
    
    def to_dict(self):
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary
        """
        return {
            'feature_extractor': self.feature_extractor,
            'max_features': self.max_features,
            'matching_algorithm': self.matching_algorithm,
            'ransac_threshold': self.ransac_threshold,
            'min_match_count': self.min_match_count,
            'index_type': self.index_type,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu,
            'use_tensor_cores': self.use_tensor_cores,
            'feature_params': self.get_feature_params(),
            'matcher_params': self.get_matcher_params(),
            'transformation_params': self.get_transformation_params(),
            'package_specific_params': self.get_package_specific_params(),
            'sheet_specific_params': self.get_sheet_specific_params(),
            'hardware_optimizations': self.optimize_for_hardware()
        }
