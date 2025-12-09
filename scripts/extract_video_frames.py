"""
Extract frames from underwater videos for BlueDepth-Crescent training
Integrates with existing project structure

Usage:
    python scripts/extract_video_frames.py
    python scripts/extract_video_frames.py --video-dir data/videos --output-dir data/frames
    python scripts/extract_video_frames.py --fps-interval 15 --max-frames 200
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_videos(
    video_dir="data/videos",
    output_dir="data/frames",
    fps_interval=30,
    max_frames_per_video=100,
    video_extensions=None
):
    """
    Extract frames from all videos in directory
    
    Args:
        video_dir (str): Directory containing video files
        output_dir (str): Output directory for extracted frames
        fps_interval (int): Extract 1 frame every N frames (30 = ~1fps for 30fps video)
        max_frames_per_video (int): Maximum frames to extract per video
        video_extensions (list): List of video file extensions to process
    
    Returns:
        int: Total number of frames extracted
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV', 
                           '.lrv', '.LRV', '.m4v', '.M4V']
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        # Also check input subfolder if it exists
        input_dir = video_dir / "input"
        if input_dir.exists():
            video_files.extend(input_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f" No videos found in {video_dir}")
        print(f"   Searched for extensions: {', '.join(video_extensions)}")
        return 0
    
    print("\n" + "="*70)
    print(f" BlueDepth-Crescent Frame Extraction")
    print("="*70)
    print(f"Found {len(video_files)} video(s)")
    print(f"Output directory: {output_dir}")
    print(f"FPS interval: Extract 1 frame every {fps_interval} frames")
    print(f"Max frames per video: {max_frames_per_video}")
    print("="*70 + "\n")
    
    total_frames_extracted = 0
    
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\n[{video_idx}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 70)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"   ERROR: Cannot open {video_path.name}")
            continue
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"   Video Info:")
        print(f"     Duration: {duration:.2f}s")
        print(f"     FPS: {video_fps:.1f}")
        print(f"     Resolution: {width}×{height}")
        print(f"     Total frames: {total_frames}")
        
        # Create output directory for this video
        video_name = video_path.stem
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        # Extract frames
        frame_idx = 0
        saved_count = 0
        
        pbar = tqdm(
            total=min(total_frames, max_frames_per_video * fps_interval),
            desc=f"    Extracting",
            leave=False,
            unit="frame"
        )
        
        while saved_count < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at intervals
            if frame_idx % fps_interval == 0:
                frame_filename = f"frame_{saved_count:04d}.jpg"
                frame_path = video_output_dir / frame_filename
                
                # Save with high quality
                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"   Extracted: {saved_count} frames")
        try:
            rel_path = video_output_dir.relative_to(Path.cwd())
            print(f"   Saved to: {rel_path}")
        except ValueError:
             print(f"   Saved to: {video_output_dir}")
        total_frames_extracted += saved_count
    
    # Summary
    print("\n" + "="*70)
    print(f" Extraction Complete!")
    print("="*70)
    print(f"Total videos processed: {len(video_files)}")
    print(f"Total frames extracted: {total_frames_extracted}")
    print(f"Average frames per video: {total_frames_extracted / len(video_files):.1f}")
    print(f"Output location: {output_dir.absolute()}")
    print("="*70 + "\n")
    
    # Next steps
    print(" Next Steps:")
    print(f"   1. Verify frames: Get-ChildItem -Recurse {output_dir} | Measure-Object")
    print(f"   2. Organize for training: python scripts\\organize_data.py")
    print(f"   3. Start training: python main.py train --model standard\n")
    
    return total_frames_extracted


def verify_opencv():
    """Verify OpenCV installation and video codec support"""
    try:
        import cv2
        print(f" OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print(" OpenCV not installed!")
        print("   Install with: pip install opencv-python")
        return False


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Extract frames from underwater videos for BlueDepth-Crescent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/videos',
        help='Directory containing video files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/frames',
        help='Output directory for extracted frames'
    )
    
    parser.add_argument(
        '--fps-interval',
        type=int,
        default=30,
        help='Extract 1 frame every N frames (30 = ~1fps for 30fps video)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum frames to extract per video'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=None,
        help='Video file extensions to process (e.g., .mp4 .avi .mov)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify OpenCV installation and exit'
    )
    
    args = parser.parse_args()
    
    # Verify OpenCV
    if args.verify or not verify_opencv():
        return 1 if not verify_opencv() else 0
    
    # Extract frames
    try:
        total_frames = extract_frames_from_videos(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            fps_interval=args.fps_interval,
            max_frames_per_video=args.max_frames,
            video_extensions=args.extensions
        )
        
        return 0 if total_frames > 0 else 1
        
    except KeyboardInterrupt:
        print("\n\n  Extraction interrupted by user")
        return 1
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
