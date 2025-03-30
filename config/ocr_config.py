"""
OCR configuration module specialized for pharmaceutical package and sheet recognition.
"""

import json
import os
from loguru import logger

class OCRConfig:
    """
    Manages OCR configuration settings with optimizations for pharmaceutical text.
    Provides specialized settings for both package and information sheet processing.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize OCR configuration with settings from config file.
        
        Args:
            config_path (str): Path to the system configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.ocr_config = config['ocr']
            self.hardware_config = config['hardware']
            self.performance_config = config['performance']
        
        # Basic OCR settings
        self.use_det = self.ocr_config.get('use_det', True)
        self.use_cls = self.ocr_config.get('use_cls', False)
        self.use_rec = self.ocr_config.get('use_rec', True)
        self.det_model_path = self.ocr_config.get('det_model_path', 'models/ocr/det_model')
        self.rec_model_path = self.ocr_config.get('rec_model_path', 'models/ocr/rec_model')
        self.cls_model_path = self.ocr_config.get('cls_model_path', '')
        self.rec_batch_num = self.ocr_config.get('rec_batch_num', 6)
        self.det_db_thresh = self.ocr_config.get('det_db_thresh', 0.3)
        self.det_db_box_thresh = self.ocr_config.get('det_db_box_thresh', 0.5)
        self.rec_char_dict_path = self.ocr_config.get('rec_char_dict_path', 'en_dict.txt')
        self.use_gpu_warmup = self.ocr_config.get('use_gpu_warmup', True)
        
        # Performance optimizations
        self.target_ocr_time_ms = self.performance_config.get('target_ocr_time_ms', 300)
        self.use_gpu = self.hardware_config.get('use_gpu', True)
        self.gpu_mem_fraction = self.hardware_config.get('gpu_memory_fraction', 0.8)
        
        # Package-specific OCR configurations
        self.package_ocr_config = {
            # High detection threshold for clear text on packages
            'det_db_thresh': 0.4,
            # Lower box threshold to catch small text on curved surfaces
            'det_db_box_thresh': 0.3,
            # Disable text angle classification for speed
            'use_angle_cls': False,
            # Smaller recognition batch for faster processing
            'rec_batch_num': 4,
            # Lower confidence threshold due to difficult package surfaces
            'min_text_confidence': 0.65,
            # Enable text direction detection for rotated text on packages
            'use_direction_classify': True,
            # Add extra preprocessing for reflective surfaces
            'preprocess_reflection_removal': True,
            # Enable contrast enhancement for faded text
            'enhance_contrast': True,
            # Special handling for curved text on blisters
            'detect_curved_text': True,
            # Additional parameters for Arabic text on packages
            'arabic_mode': True,
            # Min text size (smaller than default for package details)
            'min_text_size': 8,
            # Text clustering for repeated elements
            'use_text_clustering': True
        }
        
        # Sheet-specific OCR configurations
        self.sheet_ocr_config = {
            # Lower detection threshold for dense text documents
            'det_db_thresh': 0.2,
            # Higher box threshold for cleaner text detection
            'det_db_box_thresh': 0.55,
            # Larger batch size for dense text
            'rec_batch_num': 12,
            # Higher confidence for cleaner sheet text
            'min_text_confidence': 0.75,
            # Enable paragraph detection for information grouping
            'detect_paragraphs': True,
            # Text grouping for related information
            'group_text_lines': True,
            # Typically no need for reflection removal on sheets
            'preprocess_reflection_removal': False,
            # Enable section detection for information categorization
            'detect_sections': True,
            # Support multi-column layout processing
            'detect_columns': True,
            # Enable table detection for dosage information
            'detect_tables': True,
            # Standard text size on information sheets
            'min_text_size': 10,
            # Enable text hierarchy detection (headers, body, etc.)
            'detect_text_hierarchy': True
        }
        
        logger.info("OCR configuration initialized with specialized settings for packages and sheets")
    
    def get_base_ocr_params(self):
        """
        Get basic OCR parameters for initialization.
        
        Returns:
            dict: Base OCR parameters
        """
        return {
            'use_gpu': self.use_gpu,
            'gpu_mem': int(32 * self.gpu_mem_fraction),
            'use_tensorrt': self.use_gpu,
            'use_fp16': self.use_gpu,
            'enable_mkldnn': not self.use_gpu,
            'det_model_dir': self.det_model_path,
            'rec_model_dir': self.rec_model_path,
            'cls_model_dir': self.cls_model_path if self.use_cls else None,
            'rec_char_dict_path': self.rec_char_dict_path,
            'use_space_char': True,
            'use_angle_cls': self.use_cls,
            'det_limit_side_len': 960,
            'det_db_thresh': self.det_db_thresh,
            'det_db_box_thresh': self.det_db_box_thresh,
            'max_batch_size': self.rec_batch_num,
            'rec_batch_num': self.rec_batch_num,
            'rec_img_h': 48,
            'rec_img_w': 320,
            'use_dilation': True,
            'det_db_unclip_ratio': 1.6,
            'drop_score': 0.5
        }
    
    def get_package_ocr_params(self):
        """
        Get OCR parameters optimized for package recognition.
        
        Returns:
            dict: Package-specific OCR parameters
        """
        params = self.get_base_ocr_params()
        params.update(self.package_ocr_config)
        return params
    
    def get_sheet_ocr_params(self):
        """
        Get OCR parameters optimized for information sheet recognition.
        
        Returns:
            dict: Sheet-specific OCR parameters
        """
        params = self.get_base_ocr_params()
        params.update(self.sheet_ocr_config)
        return params
    
    def get_fast_mode_params(self):
        """
        Get ultra-fast OCR parameters for time-critical operations.
        Sacrifices some accuracy for maximum speed.
        
        Returns:
            dict: Fast-mode OCR parameters
        """
        fast_params = self.get_base_ocr_params()
        
        # Speed optimizations
        fast_params.update({
            'det_db_thresh': 0.35,  # Higher threshold = fewer detections
            'det_limit_side_len': 640,  # Smaller image size
            'det_db_unclip_ratio': 1.5,  # Less expansion of text regions
            'rec_batch_num': 16,  # Larger batch size
            'use_angle_cls': False,  # Skip rotation detection
            'drop_score': 0.65,  # Higher minimum confidence
            'min_text_size': 12,  # Ignore smaller text
            'use_dilation': False,  # Skip dilation for speed
            'unclip_ratio': 1.5  # Tighter text boxes
        })
        
        return fast_params
    
    def optimize_for_text_density(self, is_dense_text=True):
        """
        Optimize OCR parameters based on text density.
        
        Args:
            is_dense_text (bool): Whether the document has dense text
            
        Returns:
            dict: Optimized OCR parameters
        """
        if is_dense_text:
            # Dense text optimization (for information sheets)
            params = self.get_sheet_ocr_params()
            params.update({
                'det_db_thresh': 0.2,  # Lower threshold to catch more text
                'det_db_unclip_ratio': 1.8,  # Larger text regions to merge nearby text
                'use_dilation': True,  # Use dilation to connect close characters
                'group_text_lines': True  # Group text lines for paragraph processing
            })
        else:
            # Sparse text optimization (for packages)
            params = self.get_package_ocr_params()
            params.update({
                'det_db_thresh': 0.4,  # Higher threshold for clearer text
                'det_db_unclip_ratio': 1.4,  # Smaller text regions for isolated text
                'use_dilation': False,  # No dilation for isolated text
                'min_text_size': 6  # Allow smaller text detection
            })
        
        return params
    
    def optimize_for_language(self, includes_arabic=False):
        """
        Optimize OCR parameters based on language.
        
        Args:
            includes_arabic (bool): Whether the document includes Arabic text
            
        Returns:
            dict: Language-optimized OCR parameters
        """
        params = self.get_base_ocr_params()
        
        if includes_arabic:
            # Arabic text optimization
            params.update({
                'rec_char_dict_path': 'ar_dict.txt',  # Arabic dictionary
                'use_space_char': True,
                'use_direction_classify': True,  # Enable text direction detection
                'det_east_score_thresh': 0.8,  # Higher threshold for Arabic
                'det_db_unclip_ratio': 2.0,  # Larger unclip for connected scripts
                'arabic_mode': True
            })
        
        return params
