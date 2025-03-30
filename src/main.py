"""
Main application for pharmaceutical package and sheet recognition.
"""

import os
import sys
import time
import cv2
import argparse
import json
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/pharma_recognition.log", rotation="10 MB", retention="7 days", level="DEBUG")

# Import system components
from classification.document_classifier import DocumentClassifier
from template_matching.feature_matcher import FeatureMatcher
from roi.package_roi import PackageROI
from roi.sheet_roi import SheetROI
from ocr.package_processor import PackageProcessor
from ocr.sheet_processor import SheetProcessor
from utils.performance_monitor import PerformanceTimer, ComponentProfiler
from utils.image_preprocessor import preprocess_for_text_detection, resize_with_aspect_ratio
from utils.cuda_utils import CUDAUtils
from utils.validation import ValidationUtils
from db.database_connector import DatabaseConnector

class PharmaceuticalRecognitionSystem:
    """
    Main system for pharmaceutical package and sheet recognition.
    Integrates all components for end-to-end processing.
    """
    
    def __init__(self, config_path='config/system_config.json'):
        """
        Initialize the system.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.config_path = config_path
        
        # Initialize performance monitoring
        self.timer = PerformanceTimer()
        self.profiler = ComponentProfiler(enabled=True)
        
        # Initialize CUDA utilities
        self.cuda_utils = CUDAUtils(config_path)
        self.cuda_utils.optimize_memory_usage()
        self.cuda_utils.optimize_for_inference()
        self.cuda_utils.enable_tensor_cores()
        
        logger.info("Initializing pharmaceutical recognition system...")
        
        # Initialize system components
        with self.profiler.profile("system_initialization"):
            # Document classification
            self.document_classifier = DocumentClassifier(config_path)
            
            # Template matching
            self.feature_matcher = FeatureMatcher(config_path)
            
            # ROI handling
            self.package_roi = PackageROI(config_path)
            self.sheet_roi = SheetROI(config_path)
            
            # OCR processing
            self.package_processor = PackageProcessor(config_path)
            self.sheet_processor = SheetProcessor(config_path)
            
            # Validation
            self.validation = ValidationUtils(config_path)
            
            # Database connector
            self.db_connector = DatabaseConnector(config_path)
        
        # Get target performance thresholds
        performance_config = self.config.get('performance', {})
        self.max_processing_time_ms = performance_config.get('max_processing_time_ms', 500)
        
        logger.info("Pharmaceutical recognition system initialized")
    
    def process_image(self, image_path, output_path=None):
        """
        Process a pharmaceutical image.
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save results
            
        Returns:
            dict: Recognition results
        """
        # Start processing timer
        self.timer.start("total_processing_time")
        
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"error": "Failed to load image"}
        
        # Resize large images while maintaining aspect ratio
        max_dimension = 1500  # Maximum dimension for processing
        if max(image.shape[0], image.shape[1]) > max_dimension:
            image = resize_with_aspect_ratio(image, max_size=max_dimension)
        
        # Process image
        results = self.process(image)
        
        # Add image information
        results["image_path"] = image_path
        results["processing_time_ms"] = self.timer.stop("total_processing_time")
        
        # Add components performance breakdown
        results["performance"] = {
            "classification_time_ms": self.timer.get_elapsed("document_classification"),
            "template_matching_time_ms": self.timer.get_elapsed("template_matching"),
            "roi_mapping_time_ms": self.timer.get_elapsed("roi_mapping"),
            "ocr_time_ms": self.timer.get_elapsed("ocr_recognition"),
            "total_time_ms": results["processing_time_ms"]
        }
        
        # Check if processing time meets the requirement
        if results["processing_time_ms"] > self.max_processing_time_ms:
            logger.warning(f"Processing time ({results['processing_time_ms']:.2f}ms) exceeds target threshold ({self.max_processing_time_ms}ms)")
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Log processing time
        logger.info(f"Processed image in {results['processing_time_ms']:.2f}ms")
        
        # Store results in database
        if self.config.get('extraction', {}).get('use_verification', True):
            try:
                self.db_connector.add_recognition_result(results)
            except Exception as e:
                logger.error(f"Failed to store results in database: {e}")
        
        return results
    
    def process(self, image):
        """
        Process an image for pharmaceutical recognition.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Recognition results
        """
        # Apply preprocessor specific for text detection
        preprocessed = preprocess_for_text_detection(image)
        
        # Step 1: Classify document type
        with self.profiler.profile("document_classification"):
            doc_type, classification_confidence = self.document_classifier.classify(preprocessed)
            is_sheet = doc_type == DocumentClassifier.INFORMATION_SHEET
        
        logger.debug(f"Document classified as {'information sheet' if is_sheet else 'package'} with confidence {classification_confidence:.3f}")
        
        # Step 2: Match template
        with self.profiler.profile("template_matching"):
            template_id, template_type, matching_confidence, homography = self.feature_matcher.identify_template(preprocessed, is_sheet)
        
        # If template matching fails, try the other document type
        if template_id is None or matching_confidence < 0.6:
            logger.debug(f"Template matching failed, trying as {'package' if is_sheet else 'information sheet'}")
            template_id, template_type, matching_confidence, homography = self.feature_matcher.identify_template(preprocessed, not is_sheet)
            
            # Update document type if better match found
            if matching_confidence > 0.7:
                is_sheet = template_type == 'sheet'
                logger.debug(f"Reclassified as {'information sheet' if is_sheet else 'package'} based on template matching")
        
        # Track template matching result
        if template_id:
            logger.debug(f"Matched template {template_id} with confidence {matching_confidence:.3f}")
        else:
            logger.warning("No matching template found")
        
        # Initialize extraction results
        extraction_results = {}
        
        # Step 3: Extract ROIs based on document type
        if is_sheet:
            # Process as information sheet
            with self.profiler.profile("roi_mapping_sheet"):
                if template_id and homography is not None:
                    rois = self.sheet_roi.extract_field_rois(image, template_id, homography)
                else:
                    logger.warning("No valid template for ROI mapping, sheet processing may be limited")
                    rois = None
            
            # Process ROIs with sheet processor
            with self.profiler.profile("ocr_sheet"):
                if rois:
                    extraction_results = self.sheet_processor.process(image, rois)
                else:
                    # Fallback to section-based processing for sheets
                    extraction_results = self.sheet_processor.process_sections(image, None)
        else:
            # Process as package
            with self.profiler.profile("roi_mapping_package"):
                if template_id and homography is not None:
                    rois = self.package_roi.extract_field_rois(image, template_id, homography)
                else:
                    logger.warning("No valid template for ROI mapping, package processing may be limited")
                    rois = None
            
            # Process ROIs with package processor
            with self.profiler.profile("ocr_package"):
                if rois:
                    extraction_results = self.package_processor.process(image, rois)
                    
                    # Handle missing tablets for blister packs
                    if self.config.get('special_cases', {}).get('handle_missing_tablets', True):
                        package_results_with_missing = self.package_processor.process_curved_text(image, rois)
                        
                        # Merge results if better confidence
                        for field, result in package_results_with_missing.items():
                            if field in extraction_results:
                                if result.get('confidence', 0) > extraction_results[field].get('confidence', 0):
                                    extraction_results[field] = result
                else:
                    logger.warning("No ROIs available for package processing")
                    extraction_results = {}
        
        # Step 4: Validate extraction results
        with self.profiler.profile("validation"):
            validated_results = self.validation.verify_extraction_results(extraction_results)
            
            # Verify against database if available
            if self.config.get('extraction', {}).get('use_verification', True):
                verified_results = self.validation.verify_with_database(validated_results, self.db_connector)
            else:
                verified_results = validated_results
            
            # Calculate overall confidence
            overall_confidence = self.validation.calculate_overall_confidence(verified_results)
        
        # Compile final results
        results = {
            "is_sheet": is_sheet,
            "classification_confidence": classification_confidence,
            "template_id": template_id,
            "template_confidence": matching_confidence,
            "overall_confidence": overall_confidence
        }
        
        # Add extraction results
        for field, field_data in verified_results.items():
            results[field] = field_data
        
        return results
    
    def add_template(self, image_path, template_id, is_sheet=False, roi_mappings=None):
        """
        Add a new template to the system.
        
        Args:
            image_path (str): Path to template image
            template_id (str): Template ID
            is_sheet (bool): Whether it's an information sheet
            roi_mappings (dict, optional): ROI mappings for the template
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        # Resize large images while maintaining aspect ratio
        max_dimension = 1500  # Maximum dimension for processing
        if max(image.shape[0], image.shape[1]) > max_dimension:
            image = resize_with_aspect_ratio(image, max_size=max_dimension)
        
        # Add template to feature matcher
        success = self.feature_matcher.add_template(image, template_id, is_sheet)
        
        if not success:
            logger.error(f"Failed to add template: {template_id}")
            return False
        
        # Add ROI mappings if provided
        if roi_mappings:
            roi_mapper = self.sheet_roi.roi_mapper if is_sheet else self.package_roi.roi_mapper
            
            for field_name, regions in roi_mappings.items():
                roi_mapper.create_mapping(template_id, field_name, regions, is_sheet=is_sheet)
        
        logger.info(f"Added template: {template_id} ({'sheet' if is_sheet else 'package'})")
        return True
    
    def process_batch(self, image_dir, output_dir=None):
        """
        Process a batch of pharmaceutical images.
        
        Args:
            image_dir (str): Directory containing images
            output_dir (str, optional): Directory to save results
            
        Returns:
            dict: Batch processing results
        """
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        image_files = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(image_dir, filename))
        
        if not image_files:
            logger.warning(f"No image files found in {image_dir}")
            return {"error": "No image files found"}
        
        # Process each image
        batch_results = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for image_path in image_files:
            try:
                # Determine output path
                if output_dir:
                    base_name = os.path.basename(image_path)
                    name, _ = os.path.splitext(base_name)
                    output_path = os.path.join(output_dir, f"{name}_results.json")
                else:
                    output_path = None
                
                # Process image
                result = self.process_image(image_path, output_path)
                
                # Update batch results
                if "error" not in result:
                    batch_results["successful"] += 1
                else:
                    batch_results["failed"] += 1
                
                batch_results["results"].append({
                    "image_path": image_path,
                    "output_path": output_path,
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "commercial_name": result.get("commercial_name", {}).get("value"),
                    "confidence": result.get("overall_confidence", 0)
                })
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                batch_results["failed"] += 1
                batch_results["results"].append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        # Calculate performance statistics
        processing_times = [r.get("processing_time_ms", 0) for r in batch_results["results"] if "error" not in r]
        
        if processing_times:
            batch_results["performance"] = {
                "avg_processing_time_ms": sum(processing_times) / len(processing_times),
                "min_processing_time_ms": min(processing_times),
                "max_processing_time_ms": max(processing_times),
                "total_processing_time_ms": sum(processing_times)
            }
        
        # Log results
        logger.info(f"Batch processing complete: {batch_results['successful']} successful, {batch_results['failed']} failed")
        
        return batch_results
    
    def benchmark(self, image_path, iterations=10):
        """
        Benchmark system performance.
        
        Args:
            image_path (str): Path to benchmark image
            iterations (int): Number of iterations
            
        Returns:
            dict: Benchmark results
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"error": "Failed to load image"}
        
        # Resize large images while maintaining aspect ratio
        max_dimension = 1500  # Maximum dimension for processing
        if max(image.shape[0], image.shape[1]) > max_dimension:
            image = resize_with_aspect_ratio(image, max_size=max_dimension)
        
        # Reset timers
        self.timer.reset()
        
        # Run benchmark
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            result = self.process(image)
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            results.append({
                "iteration": i + 1,
                "processing_time_ms": processing_time,
                "classification_time_ms": self.timer.get_elapsed("document_classification"),
                "template_matching_time_ms": self.timer.get_elapsed("template_matching"),
                "roi_mapping_time_ms": self.timer.get_elapsed("roi_mapping"),
                "ocr_time_ms": self.timer.get_elapsed("ocr_recognition")
            })
        
        # Calculate statistics
        benchmark_results = {
            "iterations": iterations,
            "image_path": image_path,
            "results": results,
            "statistics": self.timer.get_stats()
        }
        
        # Log benchmark results
        self.timer.log_stats()
        
        return benchmark_results

def main():
    """Main function for command-line execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pharmaceutical Package and Sheet Recognition')
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--batch', help='Path to directory of images for batch processing')
    parser.add_argument('--output', help='Path to output directory')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    parser.add_argument('--config', default='config/system_config.json', help='Path to configuration file')
    parser.add_argument('--add-template', help='Add template (provide image path)')
    parser.add_argument('--template-id', help='Template ID for adding template')
    parser.add_argument('--is-sheet', action='store_true', help='Specify if template is an information sheet')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.add("logs/pharma_recognition.log", rotation="10 MB", retention="7 days", level="DEBUG")
    
    # Initialize system
    system = PharmaceuticalRecognitionSystem(args.config)
    
    # Process based on arguments
    if args.add_template and args.template_id:
        # Add template
        success = system.add_template(args.add_template, args.template_id, args.is_sheet)
        print(f"Template {'added successfully' if success else 'addition failed'}")
    elif args.image:
        # Process single image
        results = system.process_image(args.image, args.output)
        
        # Display results
        print("\nProcessing Results:")
        print(f"Document Type: {'Information Sheet' if results['is_sheet'] else 'Package'}")
        print(f"Template ID: {results.get('template_id', 'None')}")
        print(f"Processing Time: {results['processing_time_ms']:.2f}ms")
        
        for field in ['commercial_name', 'scientific_name', 'manufacturer', 'dosage']:
            if field in results:
                value = results[field].get('value', 'Not detected')
                confidence = results[field].get('confidence', 0.0)
                print(f"{field.replace('_', ' ').title()}: {value} (Confidence: {confidence:.2f})")
        
        if args.output:
            print(f"\nDetailed results saved to {args.output}")
    elif args.batch:
        # Process batch of images
        results = system.process_batch(args.batch, args.output)
        
        # Display results
        print(f"\nBatch Processing Complete:")
        print(f"Total Images: {results['total_images']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        
        if 'performance' in results:
            print(f"\nPerformance:")
            print(f"Average Processing Time: {results['performance']['avg_processing_time_ms']:.2f}ms")
            print(f"Minimum Processing Time: {results['performance']['min_processing_time_ms']:.2f}ms")
            print(f"Maximum Processing Time: {results['performance']['max_processing_time_ms']:.2f}ms")
        
        if args.output:
            print(f"\nResults saved to {args.output}")
    elif args.benchmark:
        # Run benchmark
        if not args.image:
            print("Error: Please provide an image for benchmarking with --image")
            return
        
        print(f"Running benchmark with {args.iterations} iterations...")
        results = system.benchmark(args.image, args.iterations)
        
        # Display results
        print("\nBenchmark Results:")
        print(f"Iterations: {results['iterations']}")
        
        for component, stats in results['statistics'].items():
            print(f"\n{component}:")
            print(f"  Average: {stats['mean']:.2f}ms")
            print(f"  Minimum: {stats['min']:.2f}ms")
            print(f"  Maximum: {stats['max']:.2f}ms")
            print(f"  Std Dev: {stats['std']:.2f}ms")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
