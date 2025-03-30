"""
Performance benchmarking for pharmaceutical package and sheet recognition.
"""

import os
import sys
import time
import cv2
import json
import numpy as np
import argparse
from loguru import logger
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import system components
from src.main import PharmaceuticalRecognitionSystem
from src.utils.performance_monitor import PerformanceTimer
from src.utils.image_preprocessor import resize_with_aspect_ratio
from src.config.hardware_config import HardwareConfig

class PerformanceBenchmark:
    """
    Performance benchmarking utility for the pharmaceutical recognition system.
    Provides detailed timing analysis and hardware utilization metrics.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize performance benchmark.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.timer = PerformanceTimer()
        
        # Initialize the recognition system
        self.system = PharmaceuticalRecognitionSystem(config_path)
        
        # Initialize hardware configuration
        self.hardware_config = HardwareConfig(config_path)
        
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("logs/benchmark.log", rotation="10 MB", level="DEBUG")
        
        logger.info("Performance benchmark initialized")
    
    def benchmark_single_image(self, image_path, iterations=10):
        """
        Benchmark performance on a single image.
        
        Args:
            image_path (str): Path to test image
            iterations (int): Number of iterations
            
        Returns:
            dict: Benchmark results
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"error": "Failed to load image"}
        
        # Resize large images
        image = resize_with_aspect_ratio(image, max_size=1500)
        
        # Warmup run
        logger.info("Performing warmup run...")
        self.system.process(image)
        
        # Reset timers
        self.timer.reset()
        
        # Run benchmark
        logger.info(f"Running benchmark with {iterations} iterations...")
        
        results = []
        for i in range(iterations):
            logger.debug(f"Iteration {i+1}/{iterations}")
            
            # Process image
            start_time = time.time()
            _ = self.system.process(image)
            end_time = time.time()
            
            # Calculate processing time
            processing_time_ms = (end_time - start_time) * 1000
            
            # Record component timings
            result = {
                "iteration": i + 1,
                "total_processing_time_ms": processing_time_ms,
                "classification_time_ms": self.system.timer.get_elapsed("document_classification"),
                "template_matching_time_ms": self.system.timer.get_elapsed("template_matching"),
                "roi_mapping_time_ms": self.system.timer.get_elapsed("roi_mapping"),
                "ocr_time_ms": self.system.timer.get_elapsed("ocr_recognition")
            }
            
            results.append(result)
        
        # Calculate statistics
        total_times = [r["total_processing_time_ms"] for r in results]
        classification_times = [r["classification_time_ms"] for r in results]
        template_matching_times = [r["template_matching_time_ms"] for r in results]
        roi_mapping_times = [r["roi_mapping_time_ms"] for r in results]
        ocr_times = [r["ocr_time_ms"] for r in results]
        
        stats = {
            "total_processing_time_ms": {
                "mean": np.mean(total_times),
                "median": np.median(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
                "std": np.std(total_times)
            },
            "classification_time_ms": {
                "mean": np.mean(classification_times),
                "median": np.median(classification_times),
                "min": np.min(classification_times),
                "max": np.max(classification_times),
                "std": np.std(classification_times)
            },
            "template_matching_time_ms": {
                "mean": np.mean(template_matching_times),
                "median": np.median(template_matching_times),
                "min": np.min(template_matching_times),
                "max": np.max(template_matching_times),
                "std": np.std(template_matching_times)
            },
            "roi_mapping_time_ms": {
                "mean": np.mean(roi_mapping_times),
                "median": np.median(roi_mapping_times),
                "min": np.min(roi_mapping_times),
                "max": np.max(roi_mapping_times),
                "std": np.std(roi_mapping_times)
            },
            "ocr_time_ms": {
                "mean": np.mean(ocr_times),
                "median": np.median(ocr_times),
                "min": np.min(ocr_times),
                "max": np.max(ocr_times),
                "std": np.std(ocr_times)
            }
        }
        
        # Log benchmark results
        logger.info(f"Benchmark complete. Average processing time: {stats['total_processing_time_ms']['mean']:.2f}ms")
        
        # Target threshold check
        target_time = 500  # 500ms target
        if stats['total_processing_time_ms']['mean'] > target_time:
            logger.warning(f"Average processing time ({stats['total_processing_time_ms']['mean']:.2f}ms) exceeds target ({target_time}ms)")
        else:
            logger.info(f"Performance meets target threshold of {target_time}ms")
        
        # Create benchmark report
        benchmark_results = {
            "image_path": image_path,
            "iterations": iterations,
            "results": results,
            "statistics": stats,
            "image_info": {
                "size": image.shape,
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }
        }
        
        return benchmark_results
    
    def benchmark_batch(self, image_dir, iterations=3):
        """
        Benchmark performance on a batch of images.
        
        Args:
            image_dir (str): Directory containing test images
            iterations (int): Number of iterations per image
            
        Returns:
            dict: Benchmark results
        """
        # Get list of image files
        image_files = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(image_dir, filename))
        
        if not image_files:
            logger.warning(f"No image files found in {image_dir}")
            return {"error": "No image files found"}
        
        # Run benchmark for each image
        image_results = {}
        
        for image_path in image_files:
            logger.info(f"Benchmarking {image_path}...")
            results = self.benchmark_single_image(image_path, iterations)
            image_results[os.path.basename(image_path)] = results
        
        # Calculate aggregate statistics
        all_total_times = []
        all_classification_times = []
        all_template_matching_times = []
        all_roi_mapping_times = []
        all_ocr_times = []
        
        for image_result in image_results.values():
            if "statistics" in image_result:
                all_total_times.append(image_result["statistics"]["total_processing_time_ms"]["mean"])
                all_classification_times.append(image_result["statistics"]["classification_time_ms"]["mean"])
                all_template_matching_times.append(image_result["statistics"]["template_matching_time_ms"]["mean"])
                all_roi_mapping_times.append(image_result["statistics"]["roi_mapping_time_ms"]["mean"])
                all_ocr_times.append(image_result["statistics"]["ocr_time_ms"]["mean"])
        
        aggregate_stats = {
            "total_processing_time_ms": {
                "mean": np.mean(all_total_times) if all_total_times else 0,
                "median": np.median(all_total_times) if all_total_times else 0,
                "min": np.min(all_total_times) if all_total_times else 0,
                "max": np.max(all_total_times) if all_total_times else 0,
                "std": np.std(all_total_times) if all_total_times else 0
            },
            "classification_time_ms": {
                "mean": np.mean(all_classification_times) if all_classification_times else 0,
                "median": np.median(all_classification_times) if all_classification_times else 0,
                "min": np.min(all_classification_times) if all_classification_times else 0,
                "max": np.max(all_classification_times) if all_classification_times else 0,
                "std": np.std(all_classification_times) if all_classification_times else 0
            },
            "template_matching_time_ms": {
                "mean": np.mean(all_template_matching_times) if all_template_matching_times else 0,
                "median": np.median(all_template_matching_times) if all_template_matching_times else 0,
                "min": np.min(all_template_matching_times) if all_template_matching_times else 0,
                "max": np.max(all_template_matching_times) if all_template_matching_times else 0,
                "std": np.std(all_template_matching_times) if all_template_matching_times else 0
            },
            "roi_mapping_time_ms": {
                "mean": np.mean(all_roi_mapping_times) if all_roi_mapping_times else 0,
                "median": np.median(all_roi_mapping_times) if all_roi_mapping_times else 0,
                "min": np.min(all_roi_mapping_times) if all_roi_mapping_times else 0,
                "max": np.max(all_roi_mapping_times) if all_roi_mapping_times else 0,
                "std": np.std(all_roi_mapping_times) if all_roi_mapping_times else 0
            },
            "ocr_time_ms": {
                "mean": np.mean(all_ocr_times) if all_ocr_times else 0,
                "median": np.median(all_ocr_times) if all_ocr_times else 0,
                "min": np.min(all_ocr_times) if all_ocr_times else 0,
                "max": np.max(all_ocr_times) if all_ocr_times else 0,
                "std": np.std(all_ocr_times) if all_ocr_times else 0
            }
        }
        
        # Log aggregate results
        logger.info(f"Batch benchmark complete. Average processing time across all images: {aggregate_stats['total_processing_time_ms']['mean']:.2f}ms")
        
        # Create batch report
        batch_results = {
            "image_dir": image_dir,
            "iterations_per_image": iterations,
            "total_images": len(image_files),
            "image_results": image_results,
            "aggregate_statistics": aggregate_stats
        }
        
        return batch_results
    
    def benchmark_parallel(self, image_dir, num_threads=4, iterations=3):
        """
        Benchmark parallel processing performance.
        
        Args:
            image_dir (str): Directory containing test images
            num_threads (int): Number of parallel threads
            iterations (int): Number of iterations per image
            
        Returns:
            dict: Benchmark results
        """
        # Get list of image files
        image_files = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(image_dir, filename))
        
        if not image_files:
            logger.warning(f"No image files found in {image_dir}")
            return {"error": "No image files found"}
        
        # Load images
        images = []
        for image_path in image_files:
            image = cv2.imread(image_path)
            if image is not None:
                # Resize large images
                image = resize_with_aspect_ratio(image, max_size=1500)
                images.append((os.path.basename(image_path), image))
        
        if not images:
            logger.warning("No valid images loaded")
            return {"error": "No valid images loaded"}
        
        # Warmup run
        logger.info("Performing warmup run...")
        self.system.process(images[0][1])
        
        # Run sequential benchmark first
        logger.info("Running sequential benchmark...")
        sequential_start = time.time()
        
        for _, image in images:
            for _ in range(iterations):
                self.system.process(image)
        
        sequential_time = time.time() - sequential_start
        sequential_avg_time = sequential_time * 1000 / (len(images) * iterations)
        
        # Run parallel benchmark
        logger.info(f"Running parallel benchmark with {num_threads} threads...")
        
        # Create a new system instance for each thread to avoid conflicts
        def process_image(image_data):
            name, image = image_data
            system = PharmaceuticalRecognitionSystem(self.config_path)
            results = []
            
            for i in range(iterations):
                start_time = time.time()
                _ = system.process(image)
                processing_time = (time.time() - start_time) * 1000
                
                results.append({
                    "image": name,
                    "iteration": i + 1,
                    "processing_time_ms": processing_time
                })
            
            return results
        
        parallel_start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            parallel_results = list(executor.map(process_image, images))
        
        parallel_time = time.time() - parallel_start
        parallel_avg_time = parallel_time * 1000 / (len(images) * iterations)
        
        # Flatten results
        all_results = []
        for thread_results in parallel_results:
            all_results.extend(thread_results)
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup / num_threads if num_threads > 0 else 0
        
        # Create parallel benchmark report
        parallel_benchmark = {
            "image_dir": image_dir,
            "num_threads": num_threads,
            "iterations_per_image": iterations,
            "total_images": len(images),
            "sequential_time_ms": sequential_time * 1000,
            "sequential_avg_time_ms": sequential_avg_time,
            "parallel_time_ms": parallel_time * 1000,
            "parallel_avg_time_ms": parallel_avg_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "results": all_results
        }
        
        # Log results
        logger.info(f"Parallel benchmark complete:")
        logger.info(f"  Sequential time: {sequential_time:.2f}s ({sequential_avg_time:.2f}ms per image)")
        logger.info(f"  Parallel time: {parallel_time:.2f}s ({parallel_avg_time:.2f}ms per image)")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Efficiency: {efficiency:.2f}")
        
        return parallel_benchmark
    
    def benchmark_hardware_scaling(self, image_path, cpu_thread_counts=[1, 2, 4, 6]):
        """
        Benchmark performance scaling with different hardware configurations.
        
        Args:
            image_path (str): Path to test image
            cpu_thread_counts (list): List of CPU thread counts to test
            
        Returns:
            dict: Benchmark results
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"error": "Failed to load image"}
        
        # Resize large images
        image = resize_with_aspect_ratio(image, max_size=1500)
        
        # Run benchmark for each thread count
        scaling_results = {}
        
        for thread_count in cpu_thread_counts:
            logger.info(f"Benchmarking with {thread_count} CPU threads...")
            
            # Modify environment variable for thread count
            os.environ['OMP_NUM_THREADS'] = str(thread_count)
            
            # Create a new system instance with updated thread settings
            system = PharmaceuticalRecognitionSystem(self.config_path)
            
            # Warmup run
            system.process(image)
            
            # Benchmark run
            times = []
            for i in range(5):  # 5 iterations per configuration
                start_time = time.time()
                system.process(image)
                processing_time = (time.time() - start_time) * 1000
                times.append(processing_time)
            
            # Calculate statistics
            scaling_results[thread_count] = {
                "mean_time_ms": np.mean(times),
                "median_time_ms": np.median(times),
                "min_time_ms": np.min(times),
                "max_time_ms": np.max(times),
                "std_time_ms": np.std(times),
                "times_ms": times
            }
        
        # Create hardware scaling report
        hw_scaling_results = {
            "image_path": image_path,
            "cpu_thread_counts": cpu_thread_counts,
            "results": scaling_results
        }
        
        return hw_scaling_results
    
    def visualize_benchmark(self, benchmark_results, output_path=None):
        """
        Visualize benchmark results.
        
        Args:
            benchmark_results (dict): Benchmark results
            output_path (str, optional): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Component timing breakdown
            stats = benchmark_results["statistics"]
            components = ["classification_time_ms", "template_matching_time_ms", "roi_mapping_time_ms", "ocr_time_ms"]
            component_names = ["Classification", "Template Matching", "ROI Mapping", "OCR"]
            component_means = [stats[c]["mean"] for c in components]
            
            axes[0, 0].bar(component_names, component_means)
            axes[0, 0].set_title("Component Timing Breakdown")
            axes[0, 0].set_ylabel("Time (ms)")
            
            # Add values on top of bars
            for i, v in enumerate(component_means):
                axes[0, 0].text(i, v + 1, f"{v:.1f}ms", ha="center")
            
            # Processing time distribution
            times = [r["total_processing_time_ms"] for r in benchmark_results["results"]]
            axes[0, 1].hist(times, bins=10, alpha=0.7, edgecolor="black")
            axes[0, 1].axvline(stats["total_processing_time_ms"]["mean"], color='r', linestyle='dashed', linewidth=1)
            axes[0, 1].set_title("Processing Time Distribution")
            axes[0, 1].set_xlabel("Time (ms)")
            axes[0, 1].set_ylabel("Frequency")
            
            # Iteration performance
            iterations = [r["iteration"] for r in benchmark_results["results"]]
            times = [r["total_processing_time_ms"] for r in benchmark_results["results"]]
            axes[1, 0].plot(iterations, times, marker='o')
            axes[1, 0].axhline(stats["total_processing_time_ms"]["mean"], color='r', linestyle='dashed', linewidth=1)
            axes[1, 0].set_title("Processing Time by Iteration")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Time (ms)")
            
            # Component percentage
            total_mean = stats["total_processing_time_ms"]["mean"]
            component_percentages = [c_mean / total_mean * 100 for c_mean in component_means]
            axes[1, 1].pie(component_percentages, labels=component_names, autopct='%1.1f%%')
            axes[1, 1].set_title("Component Time Percentage")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if output path provided
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Visualization saved to {output_path}")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def save_results(self, results, output_path):
        """
        Save benchmark results to file.
        
        Args:
            results (dict): Benchmark results
            output_path (str): Path to save results
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Benchmark results saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")
            return False

def main():
    """Main function for command-line execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Performance Benchmark for Pharmaceutical Recognition')
    parser.add_argument('--image', help='Path to benchmark image')
    parser.add_argument('--dir', help='Path to directory of benchmark images')
    parser.add_argument('--output', help='Path to save benchmark results')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--parallel', action='store_true', help='Run parallel benchmark')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for parallel benchmark')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--config', default='config/system_config.json', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.config)
    
    # Run benchmark
    if args.image:
        # Single image benchmark
        results = benchmark.benchmark_single_image(args.image, args.iterations)
        
        # Generate visualization
        if args.visualize:
            vis_output = os.path.splitext(args.output)[0] + ".png" if args.output else "benchmark_visualization.png"
            benchmark.visualize_benchmark(results, vis_output)
        
        # Save results
        if args.output:
            benchmark.save_results(results, args.output)
        
        # Display summary
        print("\nBenchmark Results:")
        print(f"Image: {args.image}")
        print(f"Iterations: {args.iterations}")
        print(f"Average Processing Time: {results['statistics']['total_processing_time_ms']['mean']:.2f}ms")
        print(f"Minimum Processing Time: {results['statistics']['total_processing_time_ms']['min']:.2f}ms")
        print(f"Maximum Processing Time: {results['statistics']['total_processing_time_ms']['max']:.2f}ms")
        
        # Component breakdown
        print("\nComponent Timing Breakdown:")
        total_mean = results['statistics']['total_processing_time_ms']['mean']
        
        components = [
            ("Classification", results['statistics']['classification_time_ms']['mean']),
            ("Template Matching", results['statistics']['template_matching_time_ms']['mean']),
            ("ROI Mapping", results['statistics']['roi_mapping_time_ms']['mean']),
            ("OCR", results['statistics']['ocr_time_ms']['mean'])
        ]
        
        for component_name, component_time in components:
            percentage = component_time / total_mean * 100
            print(f"  {component_name}: {component_time:.2f}ms ({percentage:.1f}%)")
    
    elif args.dir:
        if args.parallel:
            # Parallel batch benchmark
            results = benchmark.benchmark_parallel(args.dir, args.threads, args.iterations // 3)
            
            # Save results
            if args.output:
                benchmark.save_results(results, args.output)
            
            # Display summary
            print("\nParallel Benchmark Results:")
            print(f"Image Directory: {args.dir}")
            print(f"Threads: {args.threads}")
            print(f"Total Images: {results['total_images']}")
            print(f"Sequential Time: {results['sequential_time_ms']:.2f}ms ({results['sequential_avg_time_ms']:.2f}ms per image)")
            print(f"Parallel Time: {results['parallel_time_ms']:.2f}ms ({results['parallel_avg_time_ms']:.2f}ms per image)")
            print(f"Speedup: {results['speedup']:.2f}x")
            print(f"Efficiency: {results['efficiency']:.2f}")
        else:
            # Batch benchmark
            results = benchmark.benchmark_batch(args.dir, max(1, args.iterations // 3))
            
            # Save results
            if args.output:
                benchmark.save_results(results, args.output)
            
            # Display summary
            print("\nBatch Benchmark Results:")
            print(f"Image Directory: {args.dir}")
            print(f"Total Images: {results['total_images']}")
            print(f"Iterations per Image: {results['iterations_per_image']}")
            print(f"Average Processing Time: {results['aggregate_statistics']['total_processing_time_ms']['mean']:.2f}ms")
            print(f"Minimum Processing Time: {results['aggregate_statistics']['total_processing_time_ms']['min']:.2f}ms")
            print(f"Maximum Processing Time: {results['aggregate_statistics']['total_processing_time_ms']['max']:.2f}ms")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
