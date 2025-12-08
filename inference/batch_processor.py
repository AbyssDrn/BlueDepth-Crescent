"""
Batch Image Processing for Maritime Security
Processes multiple underwater images with enhancement and classification
"""

from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch process underwater images for maritime surveillance
    
    Supports enhancement, classification, and threat assessment at scale.
    """
    
    def __init__(self, enhancer, classifier=None):
        """
        Initialize batch processor
        
        Args:
            enhancer: ImageEnhancer instance
            classifier: Optional ObjectClassifier instance
        """
        self.enhancer = enhancer
        self.classifier = classifier
    
    def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        save_comparisons: bool = False,
        classify: bool = False,
        save_metadata: bool = True
    ) -> Dict[str, any]:
        """
        Process all images in directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            save_comparisons: Save before/after comparisons
            classify: Perform object classification
            save_metadata: Save processing metadata
            
        Returns:
            Processing statistics and results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in image_extensions:
            images.extend(list(input_dir.glob(ext)))
            images.extend(list(input_dir.glob(ext.upper())))
        
        if not images:
            logger.warning(f"No images found in {input_dir}")
            return {'total': 0, 'processed': 0, 'failed': 0}
        
        results = []
        processed = 0
        failed = 0
        
        # Process with progress bar
        for img_path in tqdm(images, desc="Processing images"):
            try:
                # Enhance image
                enhanced = self.enhancer.enhance_image(str(img_path))
                
                # Save enhanced image
                output_path = output_dir / img_path.name
                enhanced.save(output_path, quality=95)
                
                result = {
                    'filename': img_path.name,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Optional classification
                if classify and self.classifier:
                    classification = self.classifier.classify(enhanced)
                    result['classification'] = classification
                
                # Optional comparison
                if save_comparisons:
                    self._save_comparison(img_path, enhanced, output_dir)
                
                results.append(result)
                processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process {img_path.name}: {e}")
                results.append({
                    'filename': img_path.name,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                failed += 1
        
        # Save metadata
        if save_metadata:
            self._save_metadata(results, output_dir)
        
        stats = {
            'total': len(images),
            'processed': processed,
            'failed': failed,
            'results': results
        }
        
        logger.info(f"Batch processing complete: Processed {processed}/{len(images)} images")
        return stats
    
    def _save_comparison(self, original_path: Path, enhanced, output_dir: Path):
        """Save side-by-side comparison"""
        from PIL import Image
        
        comparison_dir = output_dir / 'comparisons'
        comparison_dir.mkdir(exist_ok=True)
        
        original = Image.open(original_path).convert('RGB')
        
        # Create comparison
        width, height = original.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(original, (0, 0))
        comparison.paste(enhanced, (width, 0))
        
        comparison_path = comparison_dir / f"comparison_{original_path.name}"
        comparison.save(comparison_path, quality=95)
    
    def _save_metadata(self, results: List[Dict], output_dir: Path):
        """Save processing metadata as JSON"""
        metadata_path = output_dir / 'processing_metadata.json'
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'model_info': self.enhancer.get_model_info(),
            'results': results
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")


class VideoFrameProcessor:
    """Process video frames for underwater surveillance"""
    
    def __init__(self, video_processor, enhancer, classifier=None):
        """
        Initialize video frame processor
        
        Args:
            video_processor: VideoProcessor instance
            enhancer: ImageEnhancer instance
            classifier: Optional ObjectClassifier instance
        """
        self.video_processor = video_processor
        self.enhancer = enhancer
        self.classifier = classifier
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        fps: int = 5,
        classify_frames: bool = False
    ) -> Dict[str, any]:
        """
        Extract frames, enhance, and optionally classify
        
        Args:
            video_path: Path to video file
            output_dir: Output directory
            fps: Frames per second to extract
            classify_frames: Perform classification on frames
            
        Returns:
            Processing results
        """
        output_dir = Path(output_dir)
        frames_dir = output_dir / 'frames'
        enhanced_dir = output_dir / 'enhanced'
        
        # Extract frames
        logger.info(f"Extracting frames from {video_path}")
        frame_paths = self.video_processor.extract_frames(
            video_path, 
            str(frames_dir), 
            fps=fps
        )
        
        # Process frames
        batch_processor = BatchProcessor(self.enhancer, self.classifier)
        results = batch_processor.process_directory(
            str(frames_dir),
            str(enhanced_dir),
            classify=classify_frames
        )
        
        # Select best frame
        best_frame = self.video_processor.select_best_frame(
            [str(enhanced_dir / Path(fp).name) for fp in frame_paths]
        )
        
        results['best_frame'] = best_frame
        results['video_path'] = video_path
        
        logger.info(f"Video processing complete: {results['processed']} frames processed")
        return results
