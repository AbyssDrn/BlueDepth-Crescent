"""Enhanced U-Net for Underwater Image Restoration"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ColorCorrectionModule(nn.Module):
    """Specialized module for underwater color correction"""
    def __init__(self, channels=64):
        super().__init__()
        self.color_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.color_branch(x)
        return x * attention

class DehazingModule(nn.Module):
    """Specialized module for underwater dehazing"""
    def __init__(self, channels=64):
        super().__init__()
        self.dehaze_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        dehaze_map = self.dehaze_branch(x)
        return x + (x * dehaze_map)

class UNetStandard(BaseModel):
    """Enhanced U-Net for Underwater Image Restoration
    
    Features:
    - Removes turbidity, haziness, blur
    - Corrects blue/green color cast
    - Improves illumination
    - Maintains natural appearance
    - Edge-preserving enhancement
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super().__init__(name="UNet-Underwater-Enhanced")
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.color_correct1 = ColorCorrectionModule(64)
        
        self.down1 = Down(64, 128)
        self.dehaze1 = DehazingModule(128)
        
        self.down2 = Down(128, 256)
        self.color_correct2 = ColorCorrectionModule(256)
        
        self.down3 = Down(256, 512)
        self.dehaze2 = DehazingModule(512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output with multi-scale refinement
        self.outc = nn.Conv2d(64, n_classes, 1)
        
        # Refinement network for final enhancement
        self.refinement = nn.Sequential(
            nn.Conv2d(n_classes, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 3, padding=1)
        )
        
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder with color correction and dehazing
        x1 = self.inc(x)
        x1 = self.color_correct1(x1)
        
        x2 = self.down1(x1)
        x2 = self.dehaze1(x2)
        
        x3 = self.down2(x2)
        x3 = self.color_correct2(x3)
        
        x4 = self.down3(x3)
        x4 = self.dehaze2(x4)
        
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Initial output
        out = self.outc(x)
        
        # Refinement for better color balance
        refined = self.refinement(out)
        final = out + refined
        
        return self.final_activation(final)