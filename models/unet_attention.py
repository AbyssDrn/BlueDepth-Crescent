"""Attention U-Net for Maximum Quality Underwater Enhancement

Best results for critical maritime security analysis

"""

import torch
import torch.nn as nn
from .base_model import BaseModel
from .unet_standard import DoubleConv, Down

class AttentionBlock(nn.Module):
    """Attention mechanism for feature refinement"""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UnderwaterAttentionModule(nn.Module):
    """Specialized attention for underwater features"""
    
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        return x * sa

class ColorEnhancementModule(nn.Module):
    """Advanced color correction for underwater scenes"""
    
    def __init__(self, channels):
        super().__init__()
        self.color_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 3, 1),
            nn.Tanh()
        )
        
        self.merge = nn.Conv2d(channels + 3, channels, 1)
    
    def forward(self, x):
        color_correction = self.color_branch(x)
        enhanced = torch.cat([x, color_correction], dim=1)
        return self.merge(enhanced)

class UNetAttention(BaseModel):
    """Attention-based U-Net for maximum quality underwater enhancement
    
    Features:
    - Attention gates for precise feature selection
    - Advanced color correction
    - Multi-scale enhancement
    - Best for high-stakes maritime security
    - Removes all underwater distortions
    
    """
    
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__(name="UNet-Attention-Maritime")
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.underwater_att1 = UnderwaterAttentionModule(64)
        
        self.down1 = Down(64, 128)
        self.color_enhance1 = ColorEnhancementModule(128)
        
        self.down2 = Down(128, 256)
        self.underwater_att2 = UnderwaterAttentionModule(256)
        
        self.down3 = Down(256, 512)
        self.color_enhance2 = ColorEnhancementModule(512)
        
        self.down4 = Down(512, 512)
        
        # Attention gates for decoder - FIXED: Matching upsampled channels
        self.att4 = AttentionBlock(512, 512, 256)  # g: 512, x: 512
        self.att3 = AttentionBlock(256, 256, 128)  # g: 256, x: 256 (FIXED from 512)
        self.att2 = AttentionBlock(128, 128, 64)   # g: 128, x: 128 (FIXED from 256)
        self.att1 = AttentionBlock(64, 64, 32)     # g: 64, x: 64 (FIXED from 128)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)
        
        # Multi-scale output
        self.outc = nn.Conv2d(64, n_classes, 1)
        
        # Final refinement
        self.final_enhance = nn.Sequential(
            nn.Conv2d(n_classes, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder with underwater-specific enhancements
        x1 = self.inc(x)              # 64 channels
        x1 = self.underwater_att1(x1)
        
        x2 = self.down1(x1)           # 128 channels
        x2 = self.color_enhance1(x2)
        
        x3 = self.down2(x2)           # 256 channels
        x3 = self.underwater_att2(x3)
        
        x4 = self.down3(x3)           # 512 channels
        x4 = self.color_enhance2(x4)
        
        x5 = self.down4(x4)           # 512 channels (bottleneck)
        
        # Decoder with attention gates
        d4 = self.up4(x5)             # 512 channels
        x4 = self.att4(d4, x4)        # Attention: g=512, x=512
        d4 = torch.cat([x4, d4], dim=1)  # 1024 channels
        d4 = self.up_conv4(d4)        # 512 channels
        
        d3 = self.up3(d4)             # 256 channels
        x3 = self.att3(d3, x3)        # Attention: g=256, x=256
        d3 = torch.cat([x3, d3], dim=1)  # 512 channels
        d3 = self.up_conv3(d3)        # 256 channels
        
        d2 = self.up2(d3)             # 128 channels
        x2 = self.att2(d2, x2)        # Attention: g=128, x=128
        d2 = torch.cat([x2, d2], dim=1)  # 256 channels
        d2 = self.up_conv2(d2)        # 128 channels
        
        d1 = self.up1(d2)             # 64 channels
        x1 = self.att1(d1, x1)        # Attention: g=64, x=64
        d1 = torch.cat([x1, d1], dim=1)  # 128 channels
        d1 = self.up_conv1(d1)        # 64 channels
        
        # Initial output
        out = self.outc(d1)
        
        # Final enhancement
        refined = self.final_enhance(out)
        final = out + refined
        
        return self.sigmoid(final)
