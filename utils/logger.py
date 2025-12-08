"""
Logging Configuration for BlueDepth-Crescent
Provides structured logging for training, inference, and debugging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog  # Install: pip install colorlog

# Log levels mapping
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = 'info',
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (auto-generated if None)
        level: Log level ('debug', 'info', 'warning', 'error')
        console_output: Enable console logging
        file_output: Enable file logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        console_formatter = CustomFormatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_file is None:
            # Auto-generate log file name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = Path('logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

class TrainingLogger:
    """Specialized logger for training with metrics tracking"""
    
    def __init__(self, name: str, log_dir: str = 'logs'):
        self.logger = setup_logger(name, log_file=f"{log_dir}/training.log")
        self.metrics_file = Path(log_dir) / 'metrics.log'
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics"""
        msg = f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)
        
        # Write to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{','.join([str(v) for v in metrics.values()])}\n")
    
    def log_checkpoint(self, epoch: int, path: str):
        """Log checkpoint save"""
        self.logger.info(f"Checkpoint saved at epoch {epoch}: {path}")
    
    def log_device_info(self, device_info: dict):
        """Log device information"""
        self.logger.info("Device Information:")
        for key, value in device_info.items():
            self.logger.info(f"  {key}: {value}")

# Module-level logger
logger = get_logger(__name__)
