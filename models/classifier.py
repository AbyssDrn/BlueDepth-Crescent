"""Complete Underwater Object Classifier with Threat Assessment
Maritime Security and Reconnaissance System
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from typing import Dict, Tuple, Optional
import numpy as np

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for feature refinement"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class UnderwaterClassifier(BaseModel):
    """Advanced classifier for underwater objects with threat assessment
    
    Maritime Security Features:
    - 15 object categories
    - 5-level threat assessment (0-4)
    - Confidence scoring
    - Real-time processing
    - Optimized for RTX 4050 and Jetson Nano
    
    Categories:
    0: Fish, 1: Coral, 2: Shark, 3: Whale, 4: Jellyfish
    5: Diver, 6: Submarine, 7: Ship, 8: Mine, 9: Debris
    10: Unknown, 11: Pipeline, 12: Cable, 13: Structure, 14: ROV
    
    Threat Levels:
    0: None, 1: Low, 2: Medium, 3: High, 4: Critical
    """
    
    def __init__(self, num_classes=15, input_size=224):
        super().__init__(name="Underwater-Maritime-Classifier")
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Feature extraction backbone with attention
        self.features = nn.ModuleList([
            # Block 1: 224x224 -> 112x112
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            
            # Block 2: 112x112 -> 56x56
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            
            # Block 3: 56x56 -> 28x28
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            
            # Block 4: 28x28 -> 14x14
            nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
        ])
        
        # Attention modules
        self.channel_attention = ChannelAttention(512)
        self.spatial_attention = SpatialAttention()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Output layers
        self.class_output = nn.Linear(128, num_classes)
        
        # Threat assessment head
        self.threat_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 5)  # 5 threat levels
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            class_logits: (B, num_classes)
            threat_logits: (B, 5)
        """
        # Feature extraction
        for block in self.features:
            x = block(x)
        
        # Apply attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        features = self.classifier(x)
        
        # Outputs
        class_logits = self.class_output(features)
        threat_logits = self.threat_head(features)
        
        return class_logits, threat_logits
    
    def predict(self, x: torch.Tensor) -> Dict[str, any]:
        """Complete prediction with confidence and threat assessment
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Dictionary with predictions, confidences, and threat levels
        """
        self.eval()
        with torch.no_grad():
            class_logits, threat_logits = self.forward(x)
            
            # Class predictions
            class_probs = F.softmax(class_logits, dim=1)
            class_confidence, predicted_class = torch.max(class_probs, dim=1)
            
            # Threat predictions
            threat_probs = F.softmax(threat_logits, dim=1)
            threat_confidence, threat_level = torch.max(threat_probs, dim=1)
            
            return {
                'class_id': predicted_class.cpu().numpy(),
                'class_name': [OBJECT_CATEGORIES[idx.item()]['name'] for idx in predicted_class],
                'class_confidence': class_confidence.cpu().numpy(),
                'class_probabilities': class_probs.cpu().numpy(),
                
                'threat_level': threat_level.cpu().numpy(),
                'threat_name': [THREAT_LEVELS[idx.item()] for idx in threat_level],
                'threat_confidence': threat_confidence.cpu().numpy(),
                'threat_probabilities': threat_probs.cpu().numpy(),
                
                'threat_description': [OBJECT_CATEGORIES[predicted_class[i].item()]['description'] 
                                     for i in range(len(predicted_class))]
            }
    
    def predict_single(self, x: torch.Tensor) -> Dict[str, any]:
        """Predict single image
        
        Args:
            x: Input tensor (1, 3, H, W) or (3, H, W)
        
        Returns:
            Single prediction dictionary
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        results = self.predict(x)
        
        # Convert to single values
        return {
            'class_id': int(results['class_id'][0]),
            'class_name': results['class_name'][0],
            'class_confidence': float(results['class_confidence'][0]),
            'threat_level': int(results['threat_level'][0]),
            'threat_name': results['threat_name'][0],
            'threat_confidence': float(results['threat_confidence'][0]),
            'threat_description': results['threat_description'][0]
        }

# Object categories with threat levels and descriptions
OBJECT_CATEGORIES = {
    0: {
        'name': 'Fish',
        'threat': 0,
        'description': 'Marine fish - No threat',
        'action': 'Monitor only'
    },
    1: {
        'name': 'Coral',
        'threat': 0,
        'description': 'Coral reef structure - No threat',
        'action': 'Navigation hazard, avoid damage'
    },
    2: {
        'name': 'Shark',
        'threat': 1,
        'description': 'Shark - Low threat (Natural behavior)',
        'action': 'Maintain safe distance, monitor'
    },
    3: {
        'name': 'Whale',
        'threat': 1,
        'description': 'Whale - Low threat',
        'action': 'Maintain safe distance, protected species'
    },
    4: {
        'name': 'Jellyfish',
        'threat': 2,
        'description': 'Jellyfish - Medium threat (Stinging capability)',
        'action': 'Avoid contact, warning to divers'
    },
    5: {
        'name': 'Diver',
        'threat': 0,
        'description': 'Human diver - No threat',
        'action': 'Identify and communicate'
    },
    6: {
        'name': 'Submarine',
        'threat': 3,
        'description': 'Submarine - High threat (Unknown intent)',
        'action': 'Alert authorities, track movement'
    },
    7: {
        'name': 'Ship',
        'threat': 2,
        'description': 'Surface vessel - Medium threat',
        'action': 'Identify vessel, monitor'
    },
    8: {
        'name': 'Mine',
        'threat': 4,
        'description': 'Underwater mine - CRITICAL THREAT',
        'action': 'IMMEDIATE EVACUATION, alert EOD team'
    },
    9: {
        'name': 'Debris',
        'threat': 1,
        'description': 'Debris - Low threat (Navigation hazard)',
        'action': 'Mark location, navigate around'
    },
    10: {
        'name': 'Unknown',
        'threat': 3,
        'description': 'Unknown object - High threat (Investigation required)',
        'action': 'Investigate with caution, document'
    },
    11: {
        'name': 'Pipeline',
        'threat': 0,
        'description': 'Underwater pipeline - Infrastructure',
        'action': 'Avoid damage, note location'
    },
    12: {
        'name': 'Cable',
        'threat': 0,
        'description': 'Underwater cable - Infrastructure',
        'action': 'Avoid damage, maintain clearance'
    },
    13: {
        'name': 'Structure',
        'threat': 1,
        'description': 'Man-made structure - Low threat',
        'action': 'Document, assess condition'
    },
    14: {
        'name': 'ROV',
        'threat': 2,
        'description': 'Remote operated vehicle - Medium threat',
        'action': 'Identify operator, monitor activity'
    }
}

# Threat level definitions
THREAT_LEVELS = {
    0: 'NONE',
    1: 'LOW',
    2: 'MEDIUM',
    3: 'HIGH',
    4: 'CRITICAL'
}

# Threat level colors for visualization
THREAT_COLORS = {
    0: (0, 255, 0),      # Green - No threat
    1: (255, 255, 0),    # Yellow - Low
    2: (255, 165, 0),    # Orange - Medium
    3: (255, 69, 0),     # Red-Orange - High
    4: (255, 0, 0)       # Red - Critical
}

def get_threat_info(class_id: int) -> Dict:
    """Get complete threat information for a class
    
    Args:
        class_id: Object class ID (0-14)
    
    Returns:
        Dictionary with threat information
    """
    if class_id not in OBJECT_CATEGORIES:
        return {
            'name': 'Unknown',
            'threat': 3,
            'description': 'Unknown object',
            'action': 'Investigate'
        }
    
    return OBJECT_CATEGORIES[class_id]

def get_threat_color(threat_level: int) -> Tuple[int, int, int]:
    """Get color code for threat level
    
    Args:
        threat_level: Threat level (0-4)
    
    Returns:
        RGB color tuple
    """
    return THREAT_COLORS.get(threat_level, (128, 128, 128))