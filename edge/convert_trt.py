#!/usr/bin/env python3
"""
Convert ONNX Models to TensorRT Engines
Optimized for NVIDIA Jetson devices
"""

import sys
import logging
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


def convert_to_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    workspace: int = 4096,
    max_batch_size: int = 1,
    verbose: bool = False
):
    """
    Convert ONNX model to TensorRT engine
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'int8')
        workspace: GPU workspace size in MB
        max_batch_size: Maximum batch size
        verbose: Enable verbose logging
    
    Requires:
        TensorRT to be installed (comes with JetPack on Jetson)
    """
    
    logger.info(f"Converting ONNX to TensorRT...")
    logger.info(f"Source: {onnx_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Workspace: {workspace} MB")
    
    # Verify ONNX file exists
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build trtexec command
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace}",
        f"--maxBatch={max_batch_size}"
    ]
    
    # Add precision flag
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
    # fp32 is default
    
    if verbose:
        cmd.append("--verbose")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run conversion
    try:
        logger.info("Starting TensorRT conversion (this may take several minutes)...")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        logger.info(" TensorRT conversion successful")
        
        if verbose:
            logger.info("STDOUT:")
            logger.info(result.stdout)
        
        # Parse output for performance info
        for line in result.stdout.split('\n'):
            if 'mean' in line.lower() or 'throughput' in line.lower():
                logger.info(line.strip())
        
    except subprocess.TimeoutExpired:
        logger.error(" Conversion timeout (>10 minutes)")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f" Conversion failed with return code {e.returncode}")
        logger.error("STDERR:")
        logger.error(e.stderr)
        raise
    except FileNotFoundError:
        logger.error(" trtexec not found!")
        logger.error("Make sure TensorRT is installed:")
        logger.error("  - On Jetson: comes with JetPack")
        logger.error("  - On x86: install tensorrt package")
        raise
    
    # Verify output file
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f" TensorRT engine saved: {output_path}")
        logger.info(f"Engine size: {file_size:.2f} MB")
    else:
        logger.error(" TensorRT engine file not created")
        raise FileNotFoundError(f"Output file not created: {output_path}")
    
    return output_path


def convert_all_models(
    onnx_dir: str = "edge/models",
    output_dir: str = "edge/models",
    precision: str = "fp16"
):
    """Convert all ONNX models to TensorRT"""
    
    onnx_dir = Path(onnx_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_files = list(onnx_dir.glob("*.onnx"))
    
    if not onnx_files:
        logger.warning(f"No ONNX files found in {onnx_dir}")
        return
    
    logger.info(f"Found {len(onnx_files)} ONNX models to convert")
    
    for onnx_file in onnx_files:
        output_file = output_dir / f"{onnx_file.stem}_{precision}.trt"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting: {onnx_file.name}")
        logger.info(f"{'='*60}")
        
        try:
            convert_to_tensorrt(
                onnx_path=str(onnx_file),
                output_path=str(output_file),
                precision=precision
            )
        except Exception as e:
            logger.error(f"Failed to convert {onnx_file.name}: {e}")


def benchmark_engine(engine_path: str, iterations: int = 100):
    """Benchmark TensorRT engine performance"""
    
    logger.info(f"Benchmarking TensorRT engine: {engine_path}")
    
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--iterations={iterations}",
        "--warmUp=200",
        "--duration=0"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse benchmark results
        for line in result.stdout.split('\n'):
            if any(keyword in line.lower() for keyword in ['throughput', 'latency', 'mean', 'percentile']):
                logger.info(line.strip())
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")


def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument('--onnx-path', type=str, help="Path to ONNX model")
    parser.add_argument('--output-path', type=str, help="Path to save TensorRT engine")
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help="Precision mode")
    parser.add_argument('--workspace', type=int, default=4096,
                       help="GPU workspace size in MB")
    parser.add_argument('--max-batch-size', type=int, default=1,
                       help="Maximum batch size")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose logging")
    parser.add_argument('--convert-all', action='store_true',
                       help="Convert all ONNX models in directory")
    parser.add_argument('--onnx-dir', type=str, default='edge/models',
                       help="Directory containing ONNX models")
    parser.add_argument('--output-dir', type=str, default='edge/models',
                       help="Directory to save TensorRT engines")
    parser.add_argument('--benchmark', type=str,
                       help="Benchmark existing TensorRT engine")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_engine(args.benchmark)
    elif args.convert_all:
        convert_all_models(args.onnx_dir, args.output_dir, args.precision)
    else:
        if not args.onnx_path or not args.output_path:
            parser.error("--onnx-path and --output-path required when not using --convert-all")
        
        convert_to_tensorrt(
            onnx_path=args.onnx_path,
            output_path=args.output_path,
            precision=args.precision,
            workspace=args.workspace,
            max_batch_size=args.max_batch_size,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
