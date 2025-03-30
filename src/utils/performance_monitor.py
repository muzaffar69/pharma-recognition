"""
Performance monitoring utilities for pharmaceutical package and sheet recognition.
"""

import time
from collections import defaultdict
import numpy as np
from loguru import logger
import os
import json
import threading

class PerformanceTimer:
    """
    Performance timer for measuring execution times of different components.
    Provides detailed timing information for optimization.
    """
    
    def __init__(self):
        """Initialize the performance timer."""
        self.timers = {}
        self.start_times = {}
        self.elapsed_times = defaultdict(list)
        self.is_running = False
        self.lock = threading.Lock()
    
    def start(self, name):
        """
        Start a timer for a specific component.
        
        Args:
            name (str): Timer name
            
        Returns:
            float: Start time in milliseconds
        """
        with self.lock:
            start_time = time.perf_counter() * 1000  # Convert to milliseconds
            self.start_times[name] = start_time
            return start_time
    
    def stop(self, name):
        """
        Stop a timer and record elapsed time.
        
        Args:
            name (str): Timer name
            
        Returns:
            float: Elapsed time in milliseconds
        """
        with self.lock:
            if name not in self.start_times:
                logger.warning(f"Timer '{name}' was not started")
                return 0.0
            
            stop_time = time.perf_counter() * 1000  # Convert to milliseconds
            elapsed_time = stop_time - self.start_times[name]
            self.elapsed_times[name].append(elapsed_time)
            
            return elapsed_time
    
    def measure(self, name):
        """
        Context manager for measuring execution time.
        
        Args:
            name (str): Timer name
            
        Returns:
            context: Context manager for measuring execution time
        """
        class TimerContext:
            def __init__(self, timer, name):
                self.timer = timer
                self.name = name
            
            def __enter__(self):
                self.timer.start(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.timer.stop(self.name)
        
        return TimerContext(self, name)
    
    def reset(self):
        """Reset all timers."""
        with self.lock:
            self.start_times = {}
            self.elapsed_times = defaultdict(list)
    
    def get_elapsed(self, name):
        """
        Get the latest elapsed time for a specific timer.
        
        Args:
            name (str): Timer name
            
        Returns:
            float: Latest elapsed time in milliseconds
        """
        with self.lock:
            if name not in self.elapsed_times or not self.elapsed_times[name]:
                return 0.0
            
            return self.elapsed_times[name][-1]
    
    def get_stats(self, name=None):
        """
        Get statistics for a specific timer or all timers.
        
        Args:
            name (str, optional): Timer name, or None for all timers
            
        Returns:
            dict: Timer statistics
        """
        with self.lock:
            if name is not None:
                if name not in self.elapsed_times:
                    return {}
                
                times = self.elapsed_times[name]
                return {
                    'count': len(times),
                    'mean': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'sum': np.sum(times),
                    'std': np.std(times)
                }
            else:
                stats = {}
                for timer_name, times in self.elapsed_times.items():
                    stats[timer_name] = {
                        'count': len(times),
                        'mean': np.mean(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'sum': np.sum(times),
                        'std': np.std(times)
                    }
                return stats
    
    def log_stats(self, name=None):
        """
        Log statistics for a specific timer or all timers.
        
        Args:
            name (str, optional): Timer name, or None for all timers
        """
        with self.lock:
            if name is not None:
                stats = self.get_stats(name)
                if stats:
                    logger.info(f"Timer '{name}': {len(self.elapsed_times[name])} runs, "
                               f"mean: {stats['mean']:.2f}ms, min: {stats['min']:.2f}ms, "
                               f"max: {stats['max']:.2f}ms, std: {stats['std']:.2f}ms")
            else:
                stats = self.get_stats()
                for timer_name, timer_stats in stats.items():
                    logger.info(f"Timer '{timer_name}': {timer_stats['count']} runs, "
                               f"mean: {timer_stats['mean']:.2f}ms, min: {timer_stats['min']:.2f}ms, "
                               f"max: {timer_stats['max']:.2f}ms, std: {timer_stats['std']:.2f}ms")
    
    def to_dict(self):
        """
        Convert timer statistics to dictionary.
        
        Returns:
            dict: Timer statistics
        """
        with self.lock:
            return self.get_stats()
    
    def to_json(self, filepath=None):
        """
        Export timer statistics to JSON.
        
        Args:
            filepath (str, optional): Output file path, or None for string
            
        Returns:
            str: JSON string if filepath is None
        """
        with self.lock:
            stats = self.get_stats()
            
            if filepath is not None:
                with open(filepath, 'w') as f:
                    json.dump(stats, f, indent=2)
            else:
                return json.dumps(stats, indent=2)
    
    def visualize(self, filepath=None):
        """
        Visualize timer statistics as a bar chart.
        
        Args:
            filepath (str, optional): Output file path
            
        Returns:
            matplotlib.figure.Figure: Figure object if available
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            with self.lock:
                stats = self.get_stats()
                
                # Extract means for each timer
                names = list(stats.keys())
                means = [stats[name]['mean'] for name in names]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create bar chart
                bars = ax.bar(np.arange(len(names)), means)
                
                # Add labels and title
                ax.set_xlabel('Component')
                ax.set_ylabel('Mean Time (ms)')
                ax.set_title('Component Performance')
                ax.set_xticks(np.arange(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.1f}ms', ha='center', va='bottom')
                
                # Adjust layout
                plt.tight_layout()
                
                # Save or show
                if filepath:
                    plt.savefig(filepath)
                
                return fig
        except ImportError:
            logger.warning("Matplotlib is required for visualization")
            return None

class ComponentProfiler:
    """
    System component profiler for optimizing performance.
    Provides detailed profiling of CPU, memory, and GPU usage.
    """
    
    def __init__(self, enabled=True):
        """
        Initialize the component profiler.
        
        Args:
            enabled (bool): Whether profiling is enabled
        """
        self.enabled = enabled
        self.timer = PerformanceTimer()
        self.cpu_usage = {}
        self.memory_usage = {}
        self.gpu_usage = {}
        
        # Try to import GPU monitoring modules
        try:
            import pynvml
            self.has_gpu_monitoring = True
            pynvml.nvmlInit()
        except (ImportError, ModuleNotFoundError):
            self.has_gpu_monitoring = False
    
    def start(self, component_name):
        """
        Start profiling a component.
        
        Args:
            component_name (str): Component name
        """
        if not self.enabled:
            return
        
        # Start timer
        self.timer.start(component_name)
        
        # Record initial resource usage
        try:
            import psutil
            self.cpu_usage[component_name] = [psutil.cpu_percent(interval=None)]
            self.memory_usage[component_name] = [psutil.Process(os.getpid()).memory_percent()]
        except (ImportError, ModuleNotFoundError):
            pass
        
        # Record GPU usage if available
        if self.has_gpu_monitoring:
            try:
                import pynvml
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_usage = []
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage.append(util.gpu)
                
                self.gpu_usage[component_name] = [gpu_usage]
            except Exception as e:
                logger.debug(f"Error recording GPU usage: {e}")
    
    def stop(self, component_name):
        """
        Stop profiling a component.
        
        Args:
            component_name (str): Component name
            
        Returns:
            dict: Profiling results
        """
        if not self.enabled:
            return {}
        
        # Stop timer
        elapsed_time = self.timer.stop(component_name)
        
        # Record final resource usage
        try:
            import psutil
            self.cpu_usage[component_name].append(psutil.cpu_percent(interval=None))
            self.memory_usage[component_name].append(psutil.Process(os.getpid()).memory_percent())
        except (ImportError, ModuleNotFoundError):
            pass
        
        # Record GPU usage if available
        if self.has_gpu_monitoring:
            try:
                import pynvml
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_usage = []
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage.append(util.gpu)
                
                self.gpu_usage[component_name].append(gpu_usage)
            except Exception as e:
                logger.debug(f"Error recording GPU usage: {e}")
        
        # Calculate resource usage statistics
        result = {
            'time_ms': elapsed_time
        }
        
        if component_name in self.cpu_usage:
            result['cpu_percent'] = np.mean(self.cpu_usage[component_name])
        
        if component_name in self.memory_usage:
            result['memory_percent'] = np.mean(self.memory_usage[component_name])
        
        if component_name in self.gpu_usage:
            gpu_means = [np.mean(usage) for usage in zip(*self.gpu_usage[component_name])]
            result['gpu_percent'] = gpu_means
        
        return result
    
    def profile(self, component_name):
        """
        Context manager for profiling a component.
        
        Args:
            component_name (str): Component name
            
        Returns:
            context: Context manager for profiling
        """
        class ProfilerContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
            
            def __enter__(self):
                self.profiler.start(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.result = self.profiler.stop(self.name)
                return False
            
            def get_result(self):
                if hasattr(self, 'result'):
                    return self.result
                return {}
        
        return ProfilerContext(self, component_name)
    
    def get_resource_usage(self):
        """
        Get current system resource usage.
        
        Returns:
            dict: Resource usage statistics
        """
        result = {}
        
        try:
            import psutil
            result['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            result['memory_percent'] = psutil.Process(os.getpid()).memory_percent()
            result['memory_mb'] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except (ImportError, ModuleNotFoundError):
            pass
        
        if self.has_gpu_monitoring:
            try:
                import pynvml
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_usage = []
                gpu_memory = []
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_usage.append(util.gpu)
                    gpu_memory.append(memory.used / memory.total * 100)
                
                result['gpu_percent'] = gpu_usage
                result['gpu_memory_percent'] = gpu_memory
            except Exception as e:
                logger.debug(f"Error getting GPU usage: {e}")
        
        return result
    
    def log_summary(self):
        """Log a summary of profiling results."""
        if not self.enabled:
            return
        
        # Get timer statistics
        timer_stats = self.timer.get_stats()
        
        # Log summary
        logger.info("=== Performance Summary ===")
        
        for component_name, stats in timer_stats.items():
            logger.info(f"Component: {component_name}")
            logger.info(f"  Time: {stats['mean']:.2f}ms (min: {stats['min']:.2f}ms, max: {stats['max']:.2f}ms)")
            
            if component_name in self.cpu_usage:
                logger.info(f"  CPU: {np.mean(self.cpu_usage[component_name]):.1f}%")
            
            if component_name in self.memory_usage:
                logger.info(f"  Memory: {np.mean(self.memory_usage[component_name]):.1f}%")
            
            if component_name in self.gpu_usage:
                gpu_means = [np.mean(usage) for usage in zip(*self.gpu_usage[component_name])]
                logger.info(f"  GPU: {gpu_means}")
            
            logger.info("")
    
    def to_json(self, filepath=None):
        """
        Export profiling results to JSON.
        
        Args:
            filepath (str, optional): Output file path, or None for string
            
        Returns:
            str: JSON string if filepath is None
        """
        if not self.enabled:
            return "{}"
        
        # Get timer statistics
        timer_stats = self.timer.get_stats()
        
        # Create profiling results
        results = {}
        
        for component_name, stats in timer_stats.items():
            results[component_name] = {
                'time_ms': {
                    'mean': stats['mean'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'std': stats['std'],
                    'count': stats['count']
                }
            }
            
            if component_name in self.cpu_usage:
                results[component_name]['cpu_percent'] = float(np.mean(self.cpu_usage[component_name]))
            
            if component_name in self.memory_usage:
                results[component_name]['memory_percent'] = float(np.mean(self.memory_usage[component_name]))
            
            if component_name in self.gpu_usage:
                gpu_means = [float(np.mean(usage)) for usage in zip(*self.gpu_usage[component_name])]
                results[component_name]['gpu_percent'] = gpu_means
        
        # Export to file or return as string
        if filepath is not None:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            return json.dumps(results, indent=2)
