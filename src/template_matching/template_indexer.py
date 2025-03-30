"""
Template indexer for efficient storage and retrieval of template features.
"""

import os
import cv2
import numpy as np
import pickle
import lmdb
import json
import shutil
from loguru import logger

class TemplateIndexer:
    """
    Manages template indexing for efficient storage and retrieval of features.
    
    Supports multiple storage types:
    - LMDB: High performance key-value storage
    - Pickle: Simple file-based storage
    - Memory: In-memory storage (fastest but not persistent)
    """
    
    def __init__(self, feature_extractor, templates_dir, index_type='lmdb'):
        """
        Initialize the template indexer.
        
        Args:
            feature_extractor: Feature extractor instance
            templates_dir (str): Directory for template storage
            index_type (str): Index storage type ('lmdb', 'pickle', 'memory')
        """
        self.feature_extractor = feature_extractor
        self.templates_dir = templates_dir
        self.index_type = index_type.lower()
        
        # Create directories if they don't exist
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Image storage directory
        self.images_dir = os.path.join(self.templates_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Feature storage
        if self.index_type == 'lmdb':
            self.db_path = os.path.join(self.templates_dir, 'templates.lmdb')
            self._init_lmdb()
        elif self.index_type == 'pickle':
            self.db_path = os.path.join(self.templates_dir, 'templates.pkl')
            self._init_pickle()
        elif self.index_type == 'memory':
            self.templates = {}
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Template indexer initialized with {self.index_type} storage at {self.templates_dir}")
    
    def _init_lmdb(self):
        """Initialize LMDB database."""
        # Open LMDB environment
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize database
        self.env = lmdb.open(self.db_path, map_size=1024*1024*1024)  # 1GB max size
    
    def _init_pickle(self):
        """Initialize pickle database."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.templates = pickle.load(f)
        else:
            self.templates = {}
            self._save_pickle()
    
    def _save_pickle(self):
        """Save template data to pickle file."""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.templates, f)
    
    def _serialize_keypoints(self, keypoints):
        """
        Serialize keypoints for storage.
        
        Args:
            keypoints (list): OpenCV keypoints
            
        Returns:
            list: Serialized keypoints
        """
        serialized = []
        for kp in keypoints:
            serialized.append({
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave,
                'class_id': kp.class_id
            })
        return serialized
    
    def _deserialize_keypoints(self, serialized):
        """
        Deserialize keypoints from storage.
        
        Args:
            serialized (list): Serialized keypoints
            
        Returns:
            list: OpenCV keypoints
        """
        keypoints = []
        for kp_data in serialized:
            kp = cv2.KeyPoint(
                x=kp_data['pt'][0],
                y=kp_data['pt'][1],
                size=kp_data['size'],
                angle=kp_data['angle'],
                response=kp_data['response'],
                octave=kp_data['octave'],
                class_id=kp_data['class_id']
            )
            keypoints.append(kp)
        return keypoints
    
    def add_template(self, template_id, keypoints, descriptors, image=None):
        """
        Add a template to the index.
        
        Args:
            template_id (str): Unique template ID
            keypoints (list): OpenCV keypoints
            descriptors (numpy.ndarray): Feature descriptors
            image (numpy.ndarray, optional): Template image for storage
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Store the image if provided
            if image is not None:
                img_path = os.path.join(self.images_dir, f"{template_id}.png")
                cv2.imwrite(img_path, image)
            
            # Serialize keypoints
            serialized_keypoints = self._serialize_keypoints(keypoints)
            
            # Create template data
            template_data = {
                'keypoints': serialized_keypoints,
                'descriptors': descriptors.tolist() if descriptors is not None else None,
                'image_path': f"{template_id}.png" if image is not None else None
            }
            
            # Store template data
            if self.index_type == 'lmdb':
                with self.env.begin(write=True) as txn:
                    txn.put(template_id.encode(), json.dumps(template_data).encode())
            elif self.index_type in ('pickle', 'memory'):
                self.templates[template_id] = template_data
                if self.index_type == 'pickle':
                    self._save_pickle()
            
            logger.info(f"Template {template_id} added to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding template {template_id}: {e}")
            return False
    
    def get_template(self, template_id):
        """
        Retrieve a template from the index.
        
        Args:
            template_id (str): Template ID
            
        Returns:
            dict: Template data with keypoints and descriptors, or None if not found
        """
        try:
            # Retrieve template data
            if self.index_type == 'lmdb':
                with self.env.begin() as txn:
                    data = txn.get(template_id.encode())
                    if data is None:
                        return None
                    template_data = json.loads(data.decode())
            elif self.index_type in ('pickle', 'memory'):
                if template_id not in self.templates:
                    return None
                template_data = self.templates[template_id]
            
            # Deserialize keypoints and convert descriptors back to numpy array
            keypoints = self._deserialize_keypoints(template_data['keypoints'])
            descriptors = np.array(template_data['descriptors'], dtype=np.float32) if template_data['descriptors'] else None
            
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image_path': os.path.join(self.images_dir, template_data['image_path']) if template_data['image_path'] else None
            }
            
        except Exception as e:
            logger.error(f"Error retrieving template {template_id}: {e}")
            return None
    
    def remove_template(self, template_id):
        """
        Remove a template from the index.
        
        Args:
            template_id (str): Template ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete image file if it exists
            img_path = os.path.join(self.images_dir, f"{template_id}.png")
            if os.path.exists(img_path):
                os.remove(img_path)
            
            # Remove template data
            if self.index_type == 'lmdb':
                with self.env.begin(write=True) as txn:
                    txn.delete(template_id.encode())
            elif self.index_type in ('pickle', 'memory'):
                if template_id in self.templates:
                    del self.templates[template_id]
                    if self.index_type == 'pickle':
                        self._save_pickle()
            
            logger.info(f"Template {template_id} removed from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing template {template_id}: {e}")
            return False
    
    def get_all_template_ids(self):
        """
        Get all template IDs in the index.
        
        Returns:
            list: List of template IDs
        """
        try:
            if self.index_type == 'lmdb':
                template_ids = []
                with self.env.begin() as txn:
                    cursor = txn.cursor()
                    for key, _ in cursor:
                        template_ids.append(key.decode())
                return template_ids
            elif self.index_type in ('pickle', 'memory'):
                return list(self.templates.keys())
            
        except Exception as e:
            logger.error(f"Error getting template IDs: {e}")
            return []
    
    def clear(self):
        """
        Clear all templates from the index.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clear image directory
            for filename in os.listdir(self.images_dir):
                file_path = os.path.join(self.images_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Clear template data
            if self.index_type == 'lmdb':
                self.env.close()
                shutil.rmtree(self.db_path)
                os.makedirs(self.db_path, exist_ok=True)
                self._init_lmdb()
            elif self.index_type in ('pickle', 'memory'):
                self.templates = {}
                if self.index_type == 'pickle':
                    self._save_pickle()
            
            logger.info("Template index cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing template index: {e}")
            return False
    
    def count(self):
        """
        Count templates in the index.
        
        Returns:
            int: Number of templates
        """
        if self.index_type == 'lmdb':
            count = 0
            with self.env.begin() as txn:
                with txn.cursor() as cursor:
                    count = sum(1 for _ in cursor)
            return count
        elif self.index_type in ('pickle', 'memory'):
            return len(self.templates)
    
    def __del__(self):
        """Clean up resources."""
        if self.index_type == 'lmdb' and hasattr(self, 'env'):
            self.env.close()
