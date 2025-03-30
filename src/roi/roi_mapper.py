"""
ROI mapper for extracting specific regions from identified templates.
"""

import os
import cv2
import numpy as np
import json
import pickle
from loguru import logger

from ..utils.performance_monitor import PerformanceTimer

class ROIMapper:
    """
    Maps regions of interest (ROIs) from templates to input images.
    Manages pre-defined extraction regions for each template.
    """
    
    # Field types for extraction
    COMMERCIAL_NAME = "commercial_name"
    SCIENTIFIC_NAME = "scientific_name"
    MANUFACTURER = "manufacturer"
    DOSAGE = "dosage"
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize the ROI mapper.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.roi_config = config['roi_mapping']
            self.paths = config['paths']
        
        # Configure parameters
        self.redundancy_factor = self.roi_config.get('redundancy_factor', 1.5)
        self.priority_levels = self.roi_config.get('priority_levels', 3)
        self.use_dynamic_adjustment = self.roi_config.get('use_dynamic_adjustment', True)
        self.max_roi_per_document = self.roi_config.get('max_roi_per_document', 20)
        
        # Mappings directories
        self.package_mapping_dir = os.path.join(self.paths['roi_mappings_dir'], 'packages')
        self.sheet_mapping_dir = os.path.join(self.paths['roi_mappings_dir'], 'sheets')
        
        # Create directories if they don't exist
        os.makedirs(self.package_mapping_dir, exist_ok=True)
        os.makedirs(self.sheet_mapping_dir, exist_ok=True)
        
        # Performance timer
        self.timer = PerformanceTimer()
        
        logger.info(f"ROI mapper initialized with redundancy factor {self.redundancy_factor}")
    
    def get_mapping_path(self, template_id, is_sheet=False):
        """
        Get the file path for a template's ROI mapping.
        
        Args:
            template_id (str): Template ID
            is_sheet (bool): Whether the template is a sheet
            
        Returns:
            str: Path to the mapping file
        """
        mapping_dir = self.sheet_mapping_dir if is_sheet else self.package_mapping_dir
        return os.path.join(mapping_dir, f"{template_id}.json")
    
    def save_mapping(self, template_id, roi_mapping, is_sheet=False):
        """
        Save ROI mapping for a template.
        
        Args:
            template_id (str): Template ID
            roi_mapping (dict): ROI mapping data
            is_sheet (bool): Whether the template is a sheet
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            mapping_path = self.get_mapping_path(template_id, is_sheet)
            
            # Convert NumPy arrays to lists for JSON serialization
            serialized_mapping = {}
            for field_name, rois in roi_mapping.items():
                serialized_rois = []
                for roi in rois:
                    serialized_roi = {
                        'region': roi['region'].tolist() if isinstance(roi['region'], np.ndarray) else roi['region'],
                        'priority': roi['priority'],
                        'description': roi.get('description', ''),
                        'ocr_config': roi.get('ocr_config', {})
                    }
                    serialized_rois.append(serialized_roi)
                serialized_mapping[field_name] = serialized_rois
            
            with open(mapping_path, 'w') as f:
                json.dump(serialized_mapping, f, indent=2)
            
            logger.info(f"ROI mapping saved for {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ROI mapping for {template_id}: {e}")
            return False
    
    def load_mapping(self, template_id, is_sheet=False):
        """
        Load ROI mapping for a template.
        
        Args:
            template_id (str): Template ID
            is_sheet (bool): Whether the template is a sheet
            
        Returns:
            dict: ROI mapping data, or None if not found
        """
        try:
            mapping_path = self.get_mapping_path(template_id, is_sheet)
            
            if not os.path.exists(mapping_path):
                logger.warning(f"No ROI mapping found for {template_id}")
                return None
            
            with open(mapping_path, 'r') as f:
                serialized_mapping = json.load(f)
            
            # Convert lists back to NumPy arrays
            roi_mapping = {}
            for field_name, serialized_rois in serialized_mapping.items():
                rois = []
                for roi in serialized_rois:
                    deserialized_roi = {
                        'region': np.array(roi['region']),
                        'priority': roi['priority'],
                        'description': roi.get('description', ''),
                        'ocr_config': roi.get('ocr_config', {})
                    }
                    rois.append(deserialized_roi)
                roi_mapping[field_name] = rois
            
            return roi_mapping
            
        except Exception as e:
            logger.error(f"Error loading ROI mapping for {template_id}: {e}")
            return None
    
    def transform_roi(self, roi, homography, target_shape):
        """
        Transform an ROI using a homography matrix.
        
        Args:
            roi (numpy.ndarray): ROI coordinates [x1, y1, x2, y2]
            homography (numpy.ndarray): Homography matrix
            target_shape (tuple): Shape of the target image (height, width)
            
        Returns:
            numpy.ndarray: Transformed ROI coordinates
        """
        # Convert ROI to corner points [top-left, top-right, bottom-right, bottom-left]
        x1, y1, x2, y2 = roi
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply homography transformation
        transformed_corners = cv2.perspectiveTransform(corners, homography)
        
        # Find bounding box of transformed ROI
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        # Apply bounds checking
        height, width = target_shape
        x_min = max(0, min(x_coords))
        y_min = max(0, min(y_coords))
        x_max = min(width - 1, max(x_coords))
        y_max = min(height - 1, max(y_coords))
        
        # Return as [x1, y1, x2, y2]
        return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)
    
    def apply_mapping(self, image, template_id, homography, is_sheet=False):
        """
        Apply ROI mapping to an image.
        
        Args:
            image (numpy.ndarray): Input image
            template_id (str): Template ID
            homography (numpy.ndarray): Homography matrix
            is_sheet (bool): Whether the template is a sheet
            
        Returns:
            dict: Extracted ROIs for each field
                {field_name: [{'image': roi_image, 'region': coordinates, 'priority': priority}, ...]}
        """
        self.timer.start("roi_mapping")
        
        # Load mapping
        roi_mapping = self.load_mapping(template_id, is_sheet)
        
        if roi_mapping is None:
            logger.warning(f"ROI mapping not found for {template_id}")
            self.timer.stop("roi_mapping")
            return None
        
        # Get target shape
        target_shape = image.shape[:2]  # (height, width)
        
        # Extract ROIs for each field
        extracted_rois = {}
        
        for field_name, rois in roi_mapping.items():
            field_rois = []
            
            for roi_data in rois:
                roi = roi_data['region']
                priority = roi_data['priority']
                description = roi_data.get('description', '')
                ocr_config = roi_data.get('ocr_config', {})
                
                # Transform ROI to target image
                try:
                    transformed_roi = self.transform_roi(roi, homography, target_shape)
                    x1, y1, x2, y2 = transformed_roi
                    
                    # Check if ROI is valid
                    if x2 <= x1 or y2 <= y1:
                        logger.debug(f"Invalid transformed ROI: {transformed_roi}")
                        continue
                    
                    # Extract ROI image
                    roi_image = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Add to extracted ROIs
                    field_rois.append({
                        'image': roi_image,
                        'region': transformed_roi,
                        'priority': priority,
                        'description': description,
                        'ocr_config': ocr_config
                    })
                    
                except Exception as e:
                    logger.error(f"Error transforming ROI: {e}")
                    continue
            
            # Sort ROIs by priority (higher first)
            field_rois.sort(key=lambda x: x['priority'], reverse=True)
            
            extracted_rois[field_name] = field_rois
        
        elapsed_ms = self.timer.stop("roi_mapping")
        logger.debug(f"ROI mapping applied for {template_id} in {elapsed_ms:.2f}ms")
        
        return extracted_rois
    
    def create_mapping(self, template_id, field_name, regions, priorities=None, descriptions=None, ocr_configs=None, is_sheet=False):
        """
        Create a new ROI mapping for a field in a template.
        
        Args:
            template_id (str): Template ID
            field_name (str): Field name (e.g., COMMERCIAL_NAME)
            regions (list): List of ROI regions [x1, y1, x2, y2]
            priorities (list, optional): List of priority values for each region
            descriptions (list, optional): List of descriptions for each region
            ocr_configs (list, optional): List of OCR configs for each region
            is_sheet (bool): Whether the template is a sheet
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load existing mapping if any
        roi_mapping = self.load_mapping(template_id, is_sheet) or {}
        
        # Default values
        num_regions = len(regions)
        if priorities is None:
            priorities = list(range(num_regions, 0, -1))
        if descriptions is None:
            descriptions = [''] * num_regions
        if ocr_configs is None:
            ocr_configs = [{}] * num_regions
        
        # Create ROIs for field
        field_rois = []
        for i in range(num_regions):
            field_rois.append({
                'region': np.array(regions[i]),
                'priority': priorities[i],
                'description': descriptions[i],
                'ocr_config': ocr_configs[i]
            })
        
        # Add to mapping
        roi_mapping[field_name] = field_rois
        
        # Save mapping
        return self.save_mapping(template_id, roi_mapping, is_sheet)
    
    def auto_adjust_mapping(self, image, template_id, homography, ocr_results, is_sheet=False):
        """
        Automatically adjust ROI mapping based on OCR results.
        
        Args:
            image (numpy.ndarray): Input image
            template_id (str): Template ID
            homography (numpy.ndarray): Homography matrix
            ocr_results (dict): OCR results for each field
            is_sheet (bool): Whether the template is a sheet
            
        Returns:
            bool: True if mapping was adjusted, False otherwise
        """
        if not self.use_dynamic_adjustment:
            return False
        
        # Load mapping
        roi_mapping = self.load_mapping(template_id, is_sheet)
        
        if roi_mapping is None:
            return False
        
        # Get target shape
        target_shape = image.shape[:2]  # (height, width)
        
        # Track if any adjustments were made
        adjusted = False
        
        # Adjust ROIs for each field
        for field_name, field_results in ocr_results.items():
            if field_name not in roi_mapping:
                continue
            
            # Only adjust if we have reliable results
            reliable_results = [r for r in field_results if r.get('confidence', 0) > 0.8]
            if not reliable_results:
                continue
            
            # Get the best result
            best_result = max(reliable_results, key=lambda x: x.get('confidence', 0))
            
            # Check if this result came from a high-priority ROI
            result_roi_index = best_result.get('roi_index', -1)
            field_rois = roi_mapping[field_name]
            
            if result_roi_index >= 0 and result_roi_index < len(field_rois):
                # If the best result came from a lower-priority ROI, increase its priority
                if field_rois[result_roi_index]['priority'] < max(r['priority'] for r in field_rois):
                    # Boost priority of successful ROI
                    field_rois[result_roi_index]['priority'] += 1
                    adjusted = True
                    logger.debug(f"Boosted priority for {field_name} ROI {result_roi_index}")
        
        # Save adjusted mapping if changes were made
        if adjusted:
            self.save_mapping(template_id, roi_mapping, is_sheet)
            logger.info(f"Adjusted ROI mapping for {template_id}")
        
        return adjusted
