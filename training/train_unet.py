"""
Main Training Script for Underwater Image Enhancement
Maritime Security and Reconnaissance System
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional

from .device_manager import DeviceManager
from .dataset import UnderwaterDataset
from .losses import CombinedLoss, PerceptualLoss
from models import UNetStandard, UNetLight, UNetAttention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNetTrainer:
    """
    Underwater Image Enhancement Trainer
    
    Features:
    - GPU thermal monitoring
    - Mixed precision training
    - TensorBoard logging
    - Automatic checkpointing
    """
    
    def __init__(
        self,
        model_type: str = 'standard',
        data_dir: str = 'data',
        img_size: int = 256,
        batch_size: int = 8,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        use_amp: bool = True,
        use_perceptual_loss: bool = False
    ):
        # Device management with thermal monitoring
        self.device_manager = DeviceManager(max_temp_celsius=80.0)
        self.device = self.device_manager.device
        
        # Adjust batch size for safety
        self.batch_size = self.device_manager.get_safe_batch_size(batch_size)
        logger.info(f"Using batch size: {self.batch_size}")
        
        # Create model
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.model = self.device_manager.optimize_for_device(self.model)
        
        # Loss functions
        self.criterion = CombinedLoss()
        self.perceptual_loss = PerceptualLoss() if use_perceptual_loss else None
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5, 
            factor=0.5,
            verbose=True
        )
        
        # Mixed precision training
        self.use_amp = use_amp and self.device_manager.using_cuda
        self.scaler = GradScaler() if self.use_amp else None
        
        # Data loading
        train_dataset = UnderwaterDataset(
            hazy_dir=f"{data_dir}/hazy",
            clear_dir=f"{data_dir}/clear",
            img_size=img_size,
            augment=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=self.device_manager.using_cuda,
            drop_last=True
        )
        
        # Training parameters
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logging
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / model_type))
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        logger.info(f"Trainer initialized: {model_type} model")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Batches per epoch: {len(self.train_loader)}")
    
    def _create_model(self, model_type: str):
        """Create model based on type"""
        if model_type == 'standard':
            return UNetStandard(n_channels=3, n_classes=3)
        elif model_type == 'light':
            return UNetLight(n_channels=3, n_classes=3)
        elif model_type == 'attention':
            return UNetAttention(n_channels=3, n_classes=3)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_epoch(self, epoch: int) -> Optional[float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, (hazy, clear) in enumerate(pbar):
            # Thermal safety check every 10 batches
            if batch_idx % 10 == 0:
                is_safe, temp = self.device_manager.check_thermal_safety()
                if not is_safe:
                    if not self.device_manager.request_thermal_continue_permission(temp):
                        logger.info("Training paused due to thermal concerns")
                        self._save_checkpoint(epoch, 'thermal_pause')
                        return None
            
            # Move data to device
            hazy = hazy.to(self.device)
            clear = clear.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(hazy)
                    loss = self.criterion(output, clear)
                    
                    # Add perceptual loss if enabled
                    if self.perceptual_loss is not None:
                        loss = loss + 0.1 * self.perceptual_loss(output, clear)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(hazy)
                loss = self.criterion(output, clear)
                
                if self.perceptual_loss is not None:
                    loss = loss + 0.1 * self.perceptual_loss(output, clear)
                
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/batch', loss.item(), self.global_step)
            
            # Clear cache periodically
            if batch_idx % 50 == 0 and self.device_manager.using_cuda:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train one epoch
            epoch_loss = self.train_epoch(epoch)
            
            if epoch_loss is None:
                # Training paused due to thermal issues
                break
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}")
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step(epoch_loss)
            
            # Save best model
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint(epoch, 'best')
                logger.info(f"Best model saved (loss: {epoch_loss:.4f})")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, f'epoch_{epoch+1}')
            
            # Memory stats
            if self.device_manager.using_cuda:
                stats = self.device_manager.get_memory_stats()
                logger.info(f"GPU Memory - Allocated: {stats['allocated']:.2f}GB")
                self.writer.add_scalar('GPU/memory_allocated', stats['allocated'], epoch)
        
        logger.info("Training completed!")
        self.writer.close()
    
    def _save_checkpoint(self, epoch: int, name: str):
        """Save model checkpoint"""
        path = self.checkpoint_dir / f"{self.model_type}_{name}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_loss,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['loss']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Resumed from epoch {self.start_epoch}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Underwater Enhancement Model")
    parser.add_argument('--model', type=str, default='standard', 
                       choices=['standard', 'light', 'attention'],
                       help='Model architecture')
    parser.add_argument('--data', type=str, default='data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = UNetTrainer(
        model_type=args.model,
        data_dir=args.data,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()
