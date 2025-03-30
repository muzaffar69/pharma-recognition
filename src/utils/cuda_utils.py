"""
CUDA utilities for GPU optimization on NVIDIA Ampere architecture.
"""

import os
import numpy as np
import paddle
from loguru import logger

class CUDAUtils:
    """
    CUDA utilities for optimizing GPU operations on NVIDIA Ampere architecture.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize CUDA utilities.
        
        Args:
            config_path (str): Path to configuration file
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.hardware_config = config['hardware']
        
        self.use_gpu = self.hardware_config.get('use_gpu', True)
        self.gpu_memory_fraction = self.hardware_config.get('gpu_memory_fraction', 0.8)
        self.tensor_cores_enabled = self.hardware_config.get('tensor_cores_enabled', True)
        
        # Check if CUDA is available
        self.cuda_available = False
        try:
            self.cuda_available = paddle.device.get_device().startswith('gpu')
        except:
            pass
        
        if self.use_gpu and not self.cuda_available:
            logger.warning("GPU acceleration requested but CUDA is not available. Falling back to CPU.")
            self.use_gpu = False
        
        logger.info(f"CUDA utils initialized (GPU available: {self.cuda_available})")
    
    def optimize_memory_usage(self):
        """Configure GPU memory usage."""
        if not self.use_gpu or not self.cuda_available:
            return
        
        try:
            # Set GPU memory fraction
            os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = str(self.gpu_memory_fraction)
            
            # Enable memory growth
            os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
            os.environ['FLAGS_fast_eager_deletion_mode'] = '1'
            
            # For PaddlePaddle compatibility
            os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
            
            logger.info(f"GPU memory usage optimized (fraction: {self.gpu_memory_fraction})")
        except Exception as e:
            logger.error(f"Error optimizing GPU memory usage: {e}")
    
    def enable_tensor_cores(self):
        """Enable Tensor Cores for optimized performance."""
        if not self.use_gpu or not self.cuda_available or not self.tensor_cores_enabled:
            return
        
        try:
            # Enable TensorCore operations
            os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP16'] = '1'
            os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP16'] = '1'
            
            # Set compute capability for Ampere
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            
            # Set paddle mixed precision
            paddle.set_default_dtype('float16')
            
            logger.info("Tensor Cores enabled for optimized performance")
        except Exception as e:
            logger.error(f"Error enabling Tensor Cores: {e}")
    
    def optimize_for_inference(self):
        """Configure GPU for optimal inference performance."""
        if not self.use_gpu or not self.cuda_available:
            return
        
        try:
            # Optimize CUDA for inference
            os.environ['CUDA_CACHE_DISABLE'] = '0'
            os.environ['CUDA_CACHE_MAXSIZE'] = '2147483647'  # Set to maximum (2GB)
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
            
            # PaddlePaddle specific optimizations
            os.environ['FLAGS_cudnn_exhaustive_search'] = '1'
            os.environ['FLAGS_cudnn_deterministic'] = '0'
            
            logger.info("GPU optimized for inference performance")
        except Exception as e:
            logger.error(f"Error optimizing GPU for inference: {e}")
    
    def optimize_for_ocr(self):
        """Configure specific optimizations for OCR workloads."""
        if not self.use_gpu or not self.cuda_available:
            return
        
        try:
            # OCR-specific optimizations
            os.environ['FLAGS_cudnn_exhaustive_search'] = '1'
            
            # Reduce CPU overhead for small kernels
            os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = '1'
            
            # Avoid unnecessary CPU-GPU syncs
            os.environ['FLAGS_max_inplace_grad_add'] = '8'
            
            logger.info("GPU optimized for OCR workloads")
        except Exception as e:
            logger.error(f"Error optimizing GPU for OCR: {e}")
    
    def get_available_memory(self):
        """
        Get available GPU memory.
        
        Returns:
            dict: GPU memory information in MB
        """
        if not self.use_gpu or not self.cuda_available:
            return {'available': 0, 'total': 0, 'used': 0}
        
        try:
            # Try using pynvml if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                return {
                    'total': info.total / (1024 * 1024),  # MB
                    'used': info.used / (1024 * 1024),    # MB
                    'available': info.free / (1024 * 1024)  # MB
                }
            except ImportError:
                pass
            
            # Fallback to PaddlePaddle's reporting
            device_info = paddle.device.cuda.get_device_properties(0)
            total_memory = device_info.total_memory
            
            # Estimate used memory (not very accurate)
            available_memory = total_memory * self.gpu_memory_fraction
            
            return {
                'total': total_memory / (1024 * 1024),  # MB
                'available': available_memory / (1024 * 1024),  # MB
                'used': (total_memory - available_memory) / (1024 * 1024)  # MB
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {'available': 0, 'total': 0, 'used': 0}
    
    def to_device(self, data):
        """
        Move data to appropriate device (GPU or CPU).
        
        Args:
            data: NumPy array or PaddlePaddle tensor
            
        Returns:
            PaddlePaddle tensor on appropriate device
        """
        if not self.use_gpu or not self.cuda_available:
            if isinstance(data, paddle.Tensor):
                return data.cpu()
            else:
                return paddle.to_tensor(data)
        
        if isinstance(data, paddle.Tensor):
            return data.cuda()
        else:
            return paddle.to_tensor(data).cuda()
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if not self.use_gpu or not self.cuda_available:
            return
        
        try:
            paddle.device.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
        except Exception as e:
            logger.error(f"Error clearing GPU memory cache: {e}")
    
    def optimize_batch_size(self, model_size_mb, input_size_mb):
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            model_size_mb (float): Model size in MB
            input_size_mb (float): Input size in MB per sample
            
        Returns:
            int: Optimal batch size
        """
        if not self.use_gpu or not self.cuda_available:
            return 1  # Default to 1 for CPU
        
        try:
            # Get available memory
            memory_info = self.get_available_memory()
            available_memory = memory_info['available']
            
            # Reserve memory for model and intermediate activations
            # Heuristic: model needs 3x its size during inference (model + activations + workspace)
            reserved_memory = model_size_mb * 3
            
            # Calculate remaining memory for inputs
            remaining_memory = max(0, available_memory - reserved_memory)
            
            # Calculate batch size
            # Each input needs approximately 2x its size during processing
            batch_size = int(remaining_memory / (input_size_mb * 2))
            
            # Ensure minimum batch size of 1
            batch_size = max(1, batch_size)
            
            # Limit batch size to powers of 2 for efficient GPU utilization
            batch_size = 2 ** int(np.log2(batch_size))
            
            logger.debug(f"Calculated optimal batch size: {batch_size} (model: {model_size_mb}MB, input: {input_size_mb}MB)")
            
            return batch_size
        except Exception as e:
            logger.error(f"Error calculating optimal batch size: {e}")
            return 1  # Default to 1 on error
    
    def estimate_memory_usage(self, height, width, channels=3, batch_size=1):
        """
        Estimate memory usage for processing an image.
        
        Args:
            height (int): Image height
            width (int): Image width
            channels (int): Number of channels
            batch_size (int): Batch size
            
        Returns:
            float: Estimated memory usage in MB
        """
        # Base image memory (assuming float32)
        image_memory = height * width * channels * 4 * batch_size / (1024 * 1024)  # MB
        
        # OCR typically needs 5-10x the image size for intermediate results
        ocr_multiplier = 8
        
        # Total estimated memory
        total_memory = image_memory * ocr_multiplier
        
        return total_memory
    
    def check_performance_mode(self):
        """
        Check if GPU is configured for optimal performance.
        
        Returns:
            dict: Performance mode information
        """
        if not self.use_gpu or not self.cuda_available:
            return {'status': 'CPU mode', 'details': {}}
        
        try:
            details = {}
            
            # Check environment variables
            details['cudnn_enabled'] = os.environ.get('FLAGS_use_cudnn', '0') == '1'
            details['tensor_cores_enabled'] = self.tensor_cores_enabled
            details['memory_fraction'] = self.gpu_memory_fraction
            
            # Check PaddlePaddle configuration
            details['default_dtype'] = str(paddle.get_default_dtype())
            
            # Evaluate status
            status = 'Optimal' if (
                details['cudnn_enabled'] and 
                details['tensor_cores_enabled'] and 
                details['memory_fraction'] >= 0.7
            ) else 'Suboptimal'
            
            return {
                'status': status,
                'details': details
            }
        except Exception as e:
            logger.error(f"Error checking performance mode: {e}")
            return {'status': 'Unknown', 'details': {}}
