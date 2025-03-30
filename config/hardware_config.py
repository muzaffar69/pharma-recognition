"""
Hardware configuration module for optimizing performance on ARM CPU and NVIDIA Ampere GPU.
"""

import os
import json
import paddle
import numpy as np
from loguru import logger

class HardwareConfig:
    """
    Hardware configuration manager for optimizing performance on ARM CPU and NVIDIA Ampere GPU.
    Provides optimization settings tailored for the 6-core ARM CPU and NVIDIA Ampere GPU architecture.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize hardware configuration with settings from config file.
        
        Args:
            config_path (str): Path to the system configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)['hardware']
        
        self.cpu_cores = self.config.get('cpu_cores', 6)
        self.gpu_memory_fraction = self.config.get('gpu_memory_fraction', 0.8)
        self.use_gpu = self.config.get('use_gpu', True)
        self.use_arm_optimization = self.config.get('use_arm_optimization', True)
        self.tensor_cores_enabled = self.config.get('tensor_cores_enabled', True)
        
        # Apply configurations
        self._configure_environment()
        self._configure_paddle()
        
        # Log configuration
        logger.info(f"Hardware configuration initialized: CPU cores={self.cpu_cores}, "
                   f"GPU enabled={self.use_gpu}, Memory fraction={self.gpu_memory_fraction}")
    
    def _configure_environment(self):
        """Configure environment variables for hardware optimization."""
        # Configure CPU threading
        os.environ['OMP_NUM_THREADS'] = str(self.cpu_cores)
        os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        
        # Configure GPU settings if available
        if self.use_gpu:
            os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = str(self.gpu_memory_fraction)
            os.environ['FLAGS_cudnn_exhaustive_search'] = '1'
            os.environ['FLAGS_cudnn_deterministic'] = '0'
            os.environ['FLAGS_use_cudnn'] = '1'
            
            # Tensor Core optimization
            if self.tensor_cores_enabled:
                os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP16'] = '1'
                os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP16'] = '1'
    
    def _configure_paddle(self):
        """Configure PaddlePaddle framework settings."""
        # Set global precision mode to float16 for performance when using Tensor Cores
        if self.use_gpu and self.tensor_cores_enabled:
            paddle.set_default_dtype('float16')
        
        # Enable GPU if available
        if self.use_gpu:
            try:
                gpu_count = paddle.device.cuda.device_count()
                if gpu_count > 0:
                    paddle.device.set_device('gpu:0')
                    logger.info(f"GPU enabled: {gpu_count} GPU(s) detected")
                else:
                    logger.warning("GPU not detected, fallback to CPU")
                    paddle.device.set_device('cpu')
                    self.use_gpu = False
            except Exception as e:
                logger.error(f"Error setting GPU device: {e}")
                paddle.device.set_device('cpu')
                self.use_gpu = False
        else:
            paddle.device.set_device('cpu')

    def optimize_numpy_operations(self):
        """
        Configure NumPy for optimal performance on ARM processors.
        Should be called before heavy numerical operations.
        """
        if self.use_arm_optimization:
            # Configure NumPy for ARM optimizations
            np.core.arrayprint._line_width = 80  # Optimize print width
            np.set_printoptions(precision=6, threshold=100, edgeitems=3)
            
            # Try to enable ARM-specific optimizations if available
            try:
                # BLAS optimizations for ARM
                if hasattr(np, '__config__') and hasattr(np.__config__, 'blas_opt_info'):
                    logger.info(f"NumPy BLAS configuration: {np.__config__.blas_opt_info}")
            except Exception as e:
                logger.warning(f"Could not configure NumPy ARM optimizations: {e}")
    
    def get_paddle_inference_config(self, model_path):
        """
        Generate optimized PaddlePaddle inference configuration.
        
        Args:
            model_path (str): Path to the model
            
        Returns:
            PaddleInferConfig: Optimized inference configuration
        """
        config = paddle.inference.Config(model_path + '.pdmodel', model_path + '.pdiparams')
        
        if self.use_gpu:
            config.enable_use_gpu(100, 0)  # 100MB initial memory, device ID 0
            config.switch_ir_optim(True)
            config.enable_memory_optim()
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=5,
                precision_mode=paddle.inference.PrecisionType.Half if self.tensor_cores_enabled else paddle.inference.PrecisionType.Float32,
                use_static=False,
                use_calib_mode=False
            )
            # Enable mixed precision
            if self.tensor_cores_enabled:
                config.enable_mixed_precision()
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.cpu_cores)
            config.enable_mkldnn()
            config.set_mkldnn_cache_capacity(10)
            
        return config
    
    def get_optimal_batch_size(self, model_size_mb):
        """
        Calculate optimal batch size based on model size and available memory.
        
        Args:
            model_size_mb (float): Model size in MB
            
        Returns:
            int: Optimal batch size
        """
        if not self.use_gpu:
            return 1  # Default to 1 for CPU
        
        # For GPU, estimate based on memory fraction
        available_gpu_memory_mb = 8 * 1024 * self.gpu_memory_fraction  # 8GB × fraction
        
        # Rough heuristic: each batch item needs ~2x model size for activations/gradients
        memory_per_item = model_size_mb * 2
        
        # Keep 20% for other operations
        available_memory = available_gpu_memory_mb * 0.8
        
        # Calculate batch size
        batch_size = max(1, int(available_memory / memory_per_item))
        
        # Limit to powers of 2 for better performance
        batch_size = 2 ** int(np.log2(batch_size))
        
        logger.info(f"Calculated optimal batch size: {batch_size} for model size {model_size_mb}MB")
        return batch_size
