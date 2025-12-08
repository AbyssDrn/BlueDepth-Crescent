"""
Video Processing Utilities for BlueDepth-Crescent
Handles video I/O, frame extraction, and video creation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List, Generator
from tqdm import tqdm

def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get video metadata
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    duration = frame_count / fps if fps > 0 else 0
    
    return {
        'fps': fps,
        'frames': frame_count,
        'duration': duration,
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}",
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }

def extract_frame(
    video_path: Union[str, Path],
    frame_number: int
) -> Optional[np.ndarray]:
    """
    Extract specific frame from video
    
    Args:
        video_path: Path to video
        frame_number: Frame index to extract
    
    Returns:
        Frame as numpy array or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def extract_frames_generator(
    video_path: Union[str, Path],
    fps: Optional[int] = None
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields frames from video
    
    Args:
        video_path: Path to video
        fps: Extract every Nth frame (None = all frames)
    
    Yields:
        (frame_number, frame) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame skip
    if fps:
        frame_skip = int(video_fps / fps)
    else:
        frame_skip = 1
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number % frame_skip == 0:
            yield frame_number, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_number += 1
    
    cap.release()

def save_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    fps: Optional[int] = 5,
    prefix: str = "frame",
    format: str = "jpg"
) -> List[Path]:
    """
    Extract and save frames from video
    
    Args:
        video_path: Path to video
        output_dir: Output directory for frames
        fps: Frames per second to extract
        prefix: Filename prefix
        format: Image format ('jpg', 'png')
    
    Returns:
        List of saved frame paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    info = get_video_info(video_path)
    saved_frames = []
    
    for frame_num, frame in tqdm(
        extract_frames_generator(video_path, fps),
        desc="Extracting frames",
        total=info['frames'] // (info['fps'] / fps) if fps else info['frames']
    ):
        frame_path = output_dir / f"{prefix}_{frame_num:06d}.{format}"
        
        # Convert RGB to BGR for cv2.imwrite
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)
        
        saved_frames.append(frame_path)
    
    return saved_frames

def create_video_from_frames(
    frame_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = 'mp4v'
) -> None:
    """
    Create video from frame images
    
    Args:
        frame_paths: List of frame image paths
        output_path: Output video path
        fps: Output video FPS
        codec: Video codec ('mp4v', 'XVID', 'H264')
    """
    if not frame_paths:
        raise ValueError("No frames provided")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame_path in tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(str(frame_path))
        out.write(frame)
    
    out.release()

class VideoWriter:
    """Context manager for video writing"""
    
    def __init__(
        self,
        output_path: Union[str, Path],
        fps: int = 30,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = 'mp4v'
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.release()
    
    def write_frame(self, frame: np.ndarray):
        """Write a single frame"""
        if self.writer is None:
            # Initialize on first frame
            height, width = frame.shape[:2]
            if self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
                width, height = self.frame_size
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (width, height)
            )
        
        # Convert RGB to BGR if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.writer.write(frame)

def calculate_video_quality(video_path: Union[str, Path]) -> dict:
    """Calculate basic video quality metrics"""
    cap = cv2.VideoCapture(str(video_path))
    
    brightness_values = []
    sharpness_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Brightness (mean intensity)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_values.append(laplacian.var())
    
    cap.release()
    
    return {
        'avg_brightness': np.mean(brightness_values),
        'avg_sharpness': np.mean(sharpness_values),
        'brightness_std': np.std(brightness_values),
        'sharpness_std': np.std(sharpness_values)
    }
