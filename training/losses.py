"""
Loss Functions for Underwater Image Enhancement
Maritime Security and Reconnaissance System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    Measures perceptual similarity between images
    """
    
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma=1.5):
        """Create Gaussian kernel"""
        x = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """Calculate SSIM loss"""
        # Move window to same device as input
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        # Calculate mean
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variance and covariance
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        # SSIM constants
        C1 = 0.01**2
        C2 = 0.03**2
        
        # Calculate SSIM
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        return 1 - ssim_map.mean(1).mean(1).mean(1)


class UnderwaterColorLoss(nn.Module):
    """
    Color correction loss for underwater images
    Penalizes bluish/greenish color dominance
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Calculate color balance loss
        Encourages proper RGB channel distribution
        """
        # Extract RGB channels
        pred_r, pred_g, pred_b = pred[:, 0], pred[:, 1], pred[:, 2]
        target_r, target_g, target_b = target[:, 0], target[:, 1], target[:, 2]
        
        # Calculate channel means
        pred_mean_r = pred_r.mean()
        pred_mean_g = pred_g.mean()
        pred_mean_b = pred_b.mean()
        
        target_mean_r = target_r.mean()
        target_mean_g = target_g.mean()
        target_mean_b = target_b.mean()
        
        # Color balance loss
        color_loss = F.l1_loss(pred_mean_r, target_mean_r) + \
                     F.l1_loss(pred_mean_g, target_mean_g) + \
                     F.l1_loss(pred_mean_b, target_mean_b)
        
        return color_loss


class EdgePreservationLoss(nn.Module):
    """
    Edge preservation loss for underwater clarity
    Maintains object boundaries and details
    """
    
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    
    def forward(self, pred, target):
        """Calculate edge preservation loss"""
        # Move filters to same device
        if self.sobel_x.device != pred.device:
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)
        
        # Calculate edges
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        # Edge loss
        return F.l1_loss(pred_edges, target_edges)


class CombinedLoss(nn.Module):
    """
    Combined loss for underwater image enhancement
    
    Includes:
    - SSIM for perceptual quality
    - L1 for pixel-level accuracy
    - Color correction for underwater distortion
    - Edge preservation for clarity
    """
    
    def __init__(
        self, 
        ssim_weight=0.5, 
        l1_weight=0.3,
        color_weight=0.1,
        edge_weight=0.1
    ):
        super().__init__()
        self.ssim = SSIMLoss()
        self.l1 = nn.L1Loss()
        self.color = UnderwaterColorLoss()
        self.edge = EdgePreservationLoss()
        
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.color_weight = color_weight
        self.edge_weight = edge_weight
    
    def forward(self, pred, target):
        """Calculate combined loss"""
        ssim_loss = self.ssim(pred, target)
        l1_loss = self.l1(pred, target)
        color_loss = self.color(pred, target)
        edge_loss = self.edge(pred, target)
        
        total_loss = (
            self.ssim_weight * ssim_loss + 
            self.l1_weight * l1_loss +
            self.color_weight * color_loss +
            self.edge_weight * edge_loss
        )
        
        return total_loss


class PerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss
    Uses pre-trained VGG16 features for perceptual similarity
    """
    
    def __init__(self):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(weights='DEFAULT').features[:16]
            self.vgg = vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        except:
            self.vgg = None
    
    def forward(self, pred, target):
        """Calculate perceptual loss"""
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Move VGG to same device
        if next(self.vgg.parameters()).device != pred.device:
            self.vgg = self.vgg.to(pred.device)
        
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        return F.mse_loss(pred_features, target_features)


class ClassificationLoss(nn.Module):
    """
    Combined loss for underwater object classification
    Includes class prediction and threat assessment
    """
    
    def __init__(self, class_weight=0.6, threat_weight=0.4):
        super().__init__()
        self.class_criterion = nn.CrossEntropyLoss()
        self.threat_criterion = nn.CrossEntropyLoss()
        self.class_weight = class_weight
        self.threat_weight = threat_weight
    
    def forward(self, class_logits, threat_logits, class_labels, threat_labels):
        """Calculate classification loss"""
        class_loss = self.class_criterion(class_logits, class_labels)
        threat_loss = self.threat_criterion(threat_logits, threat_labels)
        
        return self.class_weight * class_loss + self.threat_weight * threat_loss
