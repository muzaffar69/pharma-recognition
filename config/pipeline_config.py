"""
Pipeline configuration for pharmaceutical package and sheet recognition.
"""

import json
import os
from loguru import logger

class PipelineConfig:
    """
    Pipeline configuration for the pharmaceutical recognition system.
    Manages component-specific settings and process flow.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize pipeline configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract configuration sections
        self.performance_config = self.config.get('performance', {})
        self.hardware_config = self.config.get('hardware', {})
        self.special_cases = self.config.get('special_cases', {})
        
        # Configure pipeline stages
        self.configure_pipeline_stages()
        
        logger.info("Pipeline configuration initialized")
    
    def configure_pipeline_stages(self):
        """Configure pipeline stages based on settings."""
        # Performance targets
        self.max_processing_time_ms = self.performance_config.get('max_processing_time_ms', 500)
        self.target_classification_time_ms = self.performance_config.get('target_classification_time_ms', 50)
        self.target_template_matching_time_ms = self.performance_config.get('target_template_matching_time_ms', 100)
        self.target_ocr_time_ms = self.performance_config.get('target_ocr_time_ms', 300)
        
        # Pipeline settings
        self.parallel_processing = self.performance_config.get('parallel_processing', True)
        self.caching_enabled = self.performance_config.get('caching_enabled', True)
        self.cache_size_mb = self.performance_config.get('cache_size_mb', 512)
        
        # Special case handling
        self.handle_reflective_surfaces = self.special_cases.get('handle_reflective_surfaces', True)
        self.handle_perspective_distortion = self.special_cases.get('handle_perspective_distortion', True)
        self.detect_handwritten_notes = self.special_cases.get('detect_handwritten_notes', True)
        self.detect_stickers = self.special_cases.get('detect_stickers', True)
        self.handle_multilingual = self.special_cases.get('handle_multilingual', True)
        self.arabic_support = self.special_cases.get('arabic_support', True)
    
    def get_optimal_pipeline(self, is_speed_critical=False):
        """
        Get optimal pipeline configuration based on speed requirements.
        
        Args:
            is_speed_critical (bool): Whether speed is critical
            
        Returns:
            dict: Pipeline configuration
        """
        if is_speed_critical:
            return self._get_fast_pipeline()
        else:
            return self._get_balanced_pipeline()
    
    def _get_fast_pipeline(self):
        """
        Get pipeline configuration optimized for speed.
        
        Returns:
            dict: Fast pipeline configuration
        """
        return {
            # Classification
            'use_lightweight_classifier': True,
            
            # Template matching
            'max_template_candidates': 10,
            'min_match_count': 8,
            
            # ROI
            'max_roi_per_field': 2,
            
            # OCR
            'use_fast_mode': True,
            'skip_verification': True,
            'min_text_confidence': 0.6,
            
            # Special cases
            'handle_reflective_surfaces': False,
            'handle_perspective_distortion': False,
            'detect_handwritten_notes': False,
            'detect_stickers': False,
            'handle_multilingual': False,
            'arabic_support': False,
            
            # Performance
            'processing_timeout_ms': self.max_processing_time_ms,
        }
    
    def _get_balanced_pipeline(self):
        """
        Get pipeline configuration with balanced speed and accuracy.
        
        Returns:
            dict: Balanced pipeline configuration
        """
        return {
            # Classification
            'use_lightweight_classifier': True,
            
            # Template matching
            'max_template_candidates': 20,
            'min_match_count': 10,
            
            # ROI
            'max_roi_per_field': 3,
            
            # OCR
            'use_fast_mode': False,
            'skip_verification': False,
            'min_text_confidence': 0.7,
            
            # Special cases
            'handle_reflective_surfaces': self.handle_reflective_surfaces,
            'handle_perspective_distortion': self.handle_perspective_distortion,
            'detect_handwritten_notes': self.detect_handwritten_notes,
            'detect_stickers': self.detect_stickers,
            'handle_multilingual': self.handle_multilingual,
            'arabic_support': self.arabic_support,
            
            # Performance
            'processing_timeout_ms': self.max_processing_time_ms,
        }
    
    def get_package_pipeline(self):
        """
        Get pipeline configuration optimized for package processing.
        
        Returns:
            dict: Package pipeline configuration
        """
        # Start with balanced pipeline
        pipeline = self._get_balanced_pipeline()
        
        # Package-specific optimizations
        pipeline.update({
            'handle_reflective_surfaces': True,
            'handle_perspective_distortion': True,
            'detect_missing_tablets': True,
            'detect_curved_text': True,
            'use_adaptive_thresholding': True,
        })
        
        return pipeline
    
    def get_sheet_pipeline(self):
        """
        Get pipeline configuration optimized for sheet processing.
        
        Returns:
            dict: Sheet pipeline configuration
        """
        # Start with balanced pipeline
        pipeline = self._get_balanced_pipeline()
        
        # Sheet-specific optimizations
        pipeline.update({
            'detect_paragraphs': True,
            'group_text_lines': True,
            'detect_sections': True,
            'detect_text_hierarchy': True,
            'handle_multilingual': True,
        })
        
        return pipeline
    
    def optimize_for_hardware(self):
        """
        Optimize pipeline for specific hardware configuration.
        
        Returns:
            dict: Hardware-specific optimizations
        """
        cpu_cores = self.hardware_config.get('cpu_cores', 6)
        use_gpu = self.hardware_config.get('use_gpu', True)
        use_arm_optimization = self.hardware_config.get('use_arm_optimization', True)
        tensor_cores_enabled = self.hardware_config.get('tensor_cores_enabled', True)
        
        optimizations = {
            'parallel_processing': cpu_cores > 1,
            'parallel_threads': max(1, min(cpu_cores - 1, 4)),  # Reserve one core for main process
            'use_gpu': use_gpu,
            'use_arm_optimization': use_arm_optimization,
            'use_tensor_cores': tensor_cores_enabled,
            'batch_processing': use_gpu and cpu_cores >= 4
        }
        
        # Ampere-specific optimizations
        if use_gpu and tensor_cores_enabled:
            optimizations.update({
                'use_mixed_precision': True,
                'use_fp16': True,
                'cudnn_benchmark': True,
            })
        
        # ARM-specific optimizations
        if use_arm_optimization:
            optimizations.update({
                'thread_affinity': True,
                'simd_enabled': True,
                'optimize_memory_layout': True,
            })
        
        return optimizations
    
    def to_dict(self):
        """
        Convert pipeline configuration to dictionary.
        
        Returns:
            dict: Pipeline configuration
        """
        return {
            'max_processing_time_ms': self.max_processing_time_ms,
            'target_classification_time_ms': self.target_classification_time_ms,
            'target_template_matching_time_ms': self.target_template_matching_time_ms,
            'target_ocr_time_ms': self.target_ocr_time_ms,
            'parallel_processing': self.parallel_processing,
            'caching_enabled': self.caching_enabled,
            'cache_size_mb': self.cache_size_mb,
            'handle_reflective_surfaces': self.handle_reflective_surfaces,
            'handle_perspective_distortion': self.handle_perspective_distortion,
            'detect_handwritten_notes': self.detect_handwritten_notes,
            'detect_stickers': self.detect_stickers,
            'handle_multilingual': self.handle_multilingual,
            'arabic_support': self.arabic_support,
            'hardware_optimizations': self.optimize_for_hardware(),
            'package_pipeline': self.get_package_pipeline(),
            'sheet_pipeline': self.get_sheet_pipeline()
        }
    
    def save_config(self, path):
        """
        Save pipeline configuration to file.
        
        Args:
            path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving pipeline configuration: {e}")
            return False

def apply_pipeline_config(pipeline_config, component_config):
    """
    Apply pipeline configuration to component configuration.
    
    Args:
        pipeline_config (dict): Pipeline configuration
        component_config (dict): Component configuration
        
    Returns:
        dict: Updated component configuration
    """
    # Create a copy of the component configuration
    config = component_config.copy()
    
    # Apply pipeline settings
    for key, value in pipeline_config.items():
        if key in config:
            config[key] = value
    
    return config
