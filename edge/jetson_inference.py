#!/usr/bin/env python3
"""
TensorRT Inference Engine for NVIDIA Jetson Devices
Optimized for real-time underwater image enhancement
"""

import sys
import numpy as np
import cv2
from PIL import Image
import logging
import time
from pathlib import Path
from typing import Union, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class JetsonInference:
    """
    TensorRT inference engine for Jetson devices
    Supports FP32, FP16, and INT8 precision
    """
    
    def __init__(
        self,
        engine_path: str,
        input_size: Tuple[int, int] = (256, 256),
        warmup_iterations: int = 5
    ):
        """
        Initialize TensorRT inference engine
        
        Args:
            engine_path: Path to TensorRT engine file (.trt)
            input_size: Input image size (height, width)
            warmup_iterations: Number of warmup runs
        """
        self.engine_path = engine_path
        self.input_size = input_size
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        
        # Performance tracking
        self.total_inferences = 0
        self.total_time = 0.0
        
        # Load engine
        self._load_engine()
        self._allocate_buffers()
        self._warmup(warmup_iterations)
    
    def _load_engine(self):
        """Load TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger.info(f"Loading TensorRT engine: {self.engine_path}")
            
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Load engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Failed to load TensorRT engine")
            
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            
            logger.info(" TensorRT engine loaded successfully")
            logger.info(f"Engine: {self.engine.name}")
            logger.info(f"Max batch size: {self.engine.max_batch_size}")
            
        except ImportError as e:
            logger.error(" TensorRT or PyCUDA not installed")
            logger.error("Install with: pip install tensorrt pycuda")
            raise
        except Exception as e:
            logger.error(f" Failed to load engine: {e}")
            raise
    
    def _allocate_buffers(self):
        """Allocate CUDA buffers"""
        try:
            import pycuda.driver as cuda
            
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                    logger.info(f"Input binding: {binding}, shape: {self.engine.get_binding_shape(binding)}")
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
                    logger.info(f"Output binding: {binding}, shape: {self.engine.get_binding_shape(binding)}")
            
        except Exception as e:
            logger.error(f"Buffer allocation failed: {e}")
            raise
    
    def _warmup(self, iterations: int):
        """Warm up the engine"""
        logger.info(f"Warming up engine ({iterations} iterations)...")
        
        dummy_image = np.random.rand(*self.input_size, 3).astype(np.uint8)
        
        for i in range(iterations):
            _ = self.infer(dummy_image)
        
        logger.info(" Warmup complete")
    
    def infer(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Run inference on image
        
        Args:
            image: Input image (numpy array, file path, or PIL Image)
        
        Returns:
            Enhanced image as numpy array (H, W, 3)
        """
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image))
        
        # Preprocess
        input_data, original_shape = self._preprocess(image)
        
        # Execute inference
        output_data = self._execute(input_data)
        
        # Postprocess
        result = self._postprocess(output_data, original_shape)
        
        # Update stats
        elapsed = time.time() - start_time
        self.total_inferences += 1
        self.total_time += elapsed
        
        return result
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        Preprocess image for inference
        
        Returns:
            Preprocessed image and original shape
        """
        original_shape = image.shape
        
        # Resize to model input size
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size[::-1])  # (W, H)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Ensure contiguous array
        image = np.ascontiguousarray(image)
        
        return image, original_shape
    
    def _execute(self, input_data: np.ndarray) -> np.ndarray:
        """Execute TensorRT inference"""
        try:
            import pycuda.driver as cuda
            
            # Copy input to device
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(
                self.inputs[0]['device'],
                self.inputs[0]['host'],
                self.stream
            )
            
            # Run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            
            # Copy output to host
            cuda.memcpy_dtoh_async(
                self.outputs[0]['host'],
                self.outputs[0]['device'],
                self.stream
            )
            
            # Synchronize
            self.stream.synchronize()
            
            # Reshape output
            output = self.outputs[0]['host'].copy()
            output = output.reshape(1, 3, *self.input_size)
            
            return output
            
        except Exception as e:
            logger.error(f"Inference execution failed: {e}")
            raise
    
    def _postprocess(self, output_data: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """Postprocess model output"""
        # Remove batch dimension
        output_data = np.squeeze(output_data, axis=0)
        
        # Convert CHW to HWC
        output_data = np.transpose(output_data, (1, 2, 0))
        
        # Clip to [0, 1]
        output_data = np.clip(output_data, 0, 1)
        
        # Convert to uint8
        output_data = (output_data * 255).astype(np.uint8)
        
        # Resize back to original size
        if output_data.shape[:2] != original_shape[:2]:
            output_data = cv2.resize(
                output_data,
                (original_shape[1], original_shape[0])
            )
        
        return output_data
    
    def get_fps(self) -> float:
        """Get average FPS"""
        if self.total_inferences == 0:
            return 0.0
        return self.total_inferences / self.total_time
    
    def get_latency(self) -> float:
        """Get average latency in ms"""
        if self.total_inferences == 0:
            return 0.0
        return (self.total_time / self.total_inferences) * 1000
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_inferences = 0
        self.total_time = 0.0
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'stream') and self.stream:
            del self.stream
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jetson TensorRT inference")
    parser.add_argument('--engine', type=str, required=True,
                       help="Path to TensorRT engine")
    parser.add_argument('--input', type=str, required=True,
                       help="Input image path")
    parser.add_argument('--output', type=str, default='results/jetson_output.jpg',
                       help="Output image path")
    parser.add_argument('--benchmark', action='store_true',
                       help="Run benchmark")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = JetsonInference(args.engine)
    
    # Load and enhance image
    logger.info(f"Processing: {args.input}")
    enhanced = inference.infer(args.input)
    
    # Save result
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced).save(args.output)
    logger.info(f" Saved: {args.output}")
    
    # Print stats
    logger.info(f"Latency: {inference.get_latency():.2f} ms")
    logger.info(f"FPS: {inference.get_fps():.2f}")
    
    # Optional benchmark
    if args.benchmark:
        logger.info("\nRunning benchmark...")
        inference.reset_stats()
        
        for i in range(100):
            _ = inference.infer(args.input)
        
        logger.info(f"Average latency: {inference.get_latency():.2f} ms")
        logger.info(f"Average FPS: {inference.get_fps():.2f}")


if __name__ == "__main__":
    main()
