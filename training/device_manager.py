"""
Device Manager with Multi-GPU and Edge Device Support
Supports: Desktop GPUs, Laptop GPUs, Jetson Nano, Jetson Orin, CPU
"""

import torch
import subprocess
import logging
from typing import Optional, Tuple, Dict
import time
import platform

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Universal Device Manager for Maritime AI System
    Automatically detects and optimizes for any GPU device
    """
    
    def __init__(self, max_temp_celsius: float = 80.0, check_interval: int = 30):
        self.max_temp = max_temp_celsius
        self.check_interval = check_interval
        self.device = self._select_device()
        self.using_cuda = torch.cuda.is_available()
        self.last_temp_check = time.time()
        
        # Detect device type
        self.device_type = self._detect_device_type()
        self.device_config = self._get_device_config()
        
        logger.info(f"Device Manager initialized")
        logger.info(f"Selected device: {self.device}")
        logger.info(f"Device type: {self.device_type}")
        logger.info(f"Max temperature threshold: {self.max_temp} degrees C")
        
        if self.using_cuda:
            self._log_gpu_info()
    
    def _detect_device_type(self) -> str:
        """Detect the type of device we're running on"""
        if not self.using_cuda:
            return "CPU"
        
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            platform_info = platform.platform().lower()
            
            # Jetson devices
            if 'jetson' in platform_info or 'tegra' in gpu_name:
                if 'nano' in gpu_name or 'tegra' in gpu_name:
                    return "JETSON_NANO"
                elif 'orin' in gpu_name:
                    return "JETSON_ORIN"
                elif 'xavier' in gpu_name:
                    return "JETSON_XAVIER"
                else:
                    return "JETSON_UNKNOWN"
            
            # Desktop vs Laptop GPUs
            if 'mobile' in gpu_name or 'laptop' in gpu_name or 'max-q' in gpu_name:
                return "LAPTOP_GPU"
            else:
                return "DESKTOP_GPU"
                
        except Exception as e:
            logger.warning(f"Could not detect device type: {e}")
            return "UNKNOWN"
    
    def _get_device_config(self) -> Dict:
        """Get optimized configuration for detected device"""
        configs = {
            "JETSON_NANO": {
                "max_batch_size": 2,
                "recommended_batch_size": 1,
                "max_image_size": 256,
                "use_fp16": True,
                "memory_fraction": 0.7,
                "max_temp": 70.0,
                "description": "NVIDIA Jetson Nano (128 CUDA cores, 4GB shared RAM)"
            },
            "JETSON_ORIN": {
                "max_batch_size": 8,
                "recommended_batch_size": 4,
                "max_image_size": 512,
                "use_fp16": True,
                "memory_fraction": 0.8,
                "max_temp": 75.0,
                "description": "NVIDIA Jetson Orin"
            },
            "JETSON_XAVIER": {
                "max_batch_size": 16,
                "recommended_batch_size": 8,
                "max_image_size": 512,
                "use_fp16": True,
                "memory_fraction": 0.8,
                "max_temp": 75.0,
                "description": "NVIDIA Jetson Xavier"
            },
            "LAPTOP_GPU": {
                "max_batch_size": 8,
                "recommended_batch_size": 4,
                "max_image_size": 512,
                "use_fp16": True,
                "memory_fraction": 0.85,
                "max_temp": 80.0,
                "description": "Laptop GPU (thermal limited)"
            },
            "DESKTOP_GPU": {
                "max_batch_size": 16,
                "recommended_batch_size": 8,
                "max_image_size": 512,
                "use_fp16": True,
                "memory_fraction": 0.9,
                "max_temp": 85.0,
                "description": "Desktop GPU"
            },
            "CPU": {
                "max_batch_size": 2,
                "recommended_batch_size": 1,
                "max_image_size": 256,
                "use_fp16": False,
                "memory_fraction": 1.0,
                "max_temp": None,
                "description": "CPU only mode"
            }
        }
        
        config = configs.get(self.device_type, configs["CPU"])
        logger.info(f"Device config: {config['description']}")
        return config
    
    def _select_device(self) -> torch.device:
        """Select best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        logger.warning("WARNING: CUDA not available. CPU mode only.")
        return torch.device('cpu')
    
    def _log_gpu_info(self):
        """Log GPU information"""
        if not self.using_cuda:
            return
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_capability = torch.cuda.get_device_capability(0)
        
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Total Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Capability: {cuda_capability[0]}.{cuda_capability[1]}")
    
    def get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature"""
        if not self.using_cuda:
            return None
        
        try:
            # Try nvidia-smi for desktop/laptop GPUs
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            temp = float(result.stdout.strip())
            return temp
        except:
            # Try Jetson-specific thermal zones
            try:
                if self.device_type.startswith("JETSON"):
                    with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
                        temp = float(f.read().strip()) / 1000.0
                        return temp
            except:
                pass
        
        return None
    
    def check_thermal_safety(self, force_check: bool = False) -> Tuple[bool, Optional[float]]:
        """Check if GPU temperature is safe"""
        if not self.using_cuda:
            return True, None
        
        current_time = time.time()
        if not force_check and (current_time - self.last_temp_check) < self.check_interval:
            return True, None
        
        self.last_temp_check = current_time
        temp = self.get_gpu_temperature()
        
        if temp is None:
            return True, None
        
        # Use device-specific max temperature
        max_temp = self.device_config.get('max_temp', self.max_temp)
        is_safe = temp < max_temp
        
        if not is_safe:
            logger.error(f"CRITICAL: GPU temperature {temp:.1f}C exceeds {max_temp:.1f}C")
        elif temp > max_temp * 0.9:
            logger.warning(f"WARNING: GPU temperature high: {temp:.1f}C")
        else:
            logger.info(f"INFO: GPU temperature normal: {temp:.1f}C")
        
        return is_safe, temp
    
    def request_cpu_fallback_permission(self) -> bool:
        """Request user permission to fall back to CPU"""
        print("\n" + "="*60)
        print("GPU UNAVAILABLE OR UNSAFE")
        print("="*60)
        print("Options:")
        print("1. Switch to CPU mode (SLOW)")
        print("2. Exit and resolve GPU issues")
        print("="*60)
        
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                logger.info("User approved CPU fallback")
                return True
            elif choice == '2':
                logger.info("User declined CPU fallback")
                return False
            print("Invalid choice. Please enter 1 or 2.")
    
    def request_thermal_continue_permission(self, current_temp: float) -> bool:
        """Request permission to continue despite high temperature"""
        max_temp = self.device_config.get('max_temp', self.max_temp)
        
        print("\n" + "="*60)
        print(f"HIGH GPU TEMPERATURE: {current_temp:.1f}C")
        print(f"Threshold: {max_temp:.1f}C")
        print(f"Device: {self.device_type}")
        print("="*60)
        print("WARNING: Continuing may damage your hardware!")
        print("Options:")
        print("1. Pause training and cool down (RECOMMENDED)")
        print("2. Continue anyway (DANGEROUS)")
        print("3. Switch to CPU mode")
        print("="*60)
        
        while True:
            choice = input("Enter choice (1, 2, or 3): ").strip()
            if choice == '1':
                return False
            elif choice == '2':
                confirm = input("Type 'CONFIRM' to continue: ").strip()
                if confirm == 'CONFIRM':
                    logger.warning("User forced continue despite thermal warning")
                    return True
            elif choice == '3':
                if self.request_cpu_fallback_permission():
                    self.device = torch.device('cpu')
                    self.using_cuda = False
                    return True
                return False
            print("Invalid choice.")
    
    def get_safe_batch_size(self, base_batch_size: int = 8) -> int:
        """Get safe batch size based on device capabilities"""
        if not self.using_cuda:
            return max(1, base_batch_size // 4)
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            recommended_batch = self.device_config['recommended_batch_size']
            max_batch = self.device_config['max_batch_size']
            
            # For Jetson Nano, be very conservative
            if self.device_type == "JETSON_NANO":
                return min(recommended_batch, base_batch_size)
            
            # Adjust based on available memory
            if total_memory < 4:
                return min(2, base_batch_size)
            elif total_memory < 6:
                return min(4, base_batch_size)
            elif total_memory < 8:
                return min(recommended_batch, base_batch_size)
            else:
                return min(max_batch, base_batch_size)
                
        except Exception as e:
            logger.warning(f"Error determining batch size: {e}")
            return self.device_config['recommended_batch_size']
    
    def get_optimal_image_size(self) -> int:
        """Get optimal image size for device"""
        return self.device_config['max_image_size']
    
    def should_use_fp16(self) -> bool:
        """Check if FP16 (half precision) should be used"""
        if not self.using_cuda:
            return False
        return self.device_config['use_fp16']
    
    def optimize_for_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for current device"""
        model = model.to(self.device)
        
        if self.using_cuda:
            logger.info(f"Model moved to {self.device_type}")
            
            # Enable cudnn benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            
            # For Jetson, additional optimizations
            if self.device_type.startswith("JETSON"):
                logger.info("Applying Jetson-specific optimizations")
                if torch.cuda.get_device_capability()[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
        else:
            logger.info("Model running on CPU")
        
        return model
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not self.using_cuda:
            return {}
        
        stats = {
            'allocated': torch.cuda.memory_allocated(0) / 1024**3,
            'reserved': torch.cuda.memory_reserved(0) / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated(0) / 1024**3,
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
        
        stats['utilization_percent'] = (stats['allocated'] / stats['total']) * 100
        return stats
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.using_cuda:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def get_device_info(self) -> Dict:
        """Get comprehensive device information"""
        info = {
            'device_type': self.device_type,
            'device': str(self.device),
            'cuda_available': self.using_cuda,
            'config': self.device_config
        }
        
        if self.using_cuda:
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            })
        
        return info
