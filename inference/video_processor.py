"""
Video Frame Extraction and Processing for Maritime Surveillance
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Extract and process frames from underwater surveillance videos
    """
    
    def __init__(self, max_duration: int = 60):
        """
        Initialize video processor
        
        Args:
            max_duration: Maximum video duration to process (seconds)
        """
        self.max_duration = max_duration
    
    def extract_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        fps: int = 5
    ) -> List[str]:
        """
        Extract frames from video at specified fps
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract
            
        Returns:
            List of extracted frame paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        if duration > self.max_duration:
            logger.warning(
                f"Video duration {duration:.1f}s exceeds max {self.max_duration}s. "
                f"Processing first {self.max_duration}s only."
            )
        
        frame_interval = int(video_fps / fps) if video_fps > 0 else 1
        extracted = []
        frame_idx = 0
        saved_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Stop if exceeding max duration
            if (frame_idx / video_fps) > self.max_duration:
                break
            
            if frame_idx % frame_interval == 0:
                output_path = output_dir / f"frame_{saved_idx:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                extracted.append(str(output_path))
                saved_idx += 1
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(extracted)} frames from {video_path.name}")
        
        return extracted
    
    def select_best_frame(self, frames: List[str]) -> Optional[str]:
        """
        Select best frame based on quality metrics
        
        Args:
            frames: List of frame paths
            
        Returns:
            Path to best frame
        """
        if not frames:
            return None
        
        best_frame = None
        best_score = -1
        
        for frame_path in frames:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            
            score = self._calculate_quality_score(img)
            
            if score > best_score:
                best_score = score
                best_frame = frame_path
        
        if best_frame:
            logger.info(f"Best frame: {Path(best_frame).name} (score: {best_score:.2f})")
        
        return best_frame
    
    def _calculate_quality_score(self, img: np.ndarray) -> float:
        """
        Calculate image quality score based on sharpness, brightness, and contrast
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Quality score (higher is better)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Brightness (prefer middle range)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        # Contrast
        contrast = gray.std()
        
        # Combined score with weights
        score = (sharpness * 0.5) + (brightness_score * 0.3) + (contrast * 0.2)
        
        return score
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
