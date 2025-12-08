"""
UI Components for BlueDepth-Crescent Dashboard
Professional modular components
"""

from .header import create_header
from .model_selector import ModelSelector
from .metrics_panel import create_metrics_panel, create_metrics_html, calculate_all_metrics
from .comparison_panel import create_comparison_panel, create_comparison_image
from .charts_panel import create_charts_panel, create_training_chart
from .classification_panel import create_classification_panel, format_classification_html
from .upload_panel import create_upload_panel

__all__ = [
    'create_header',
    'ModelSelector',
    'create_metrics_panel',
    'create_metrics_html',
    'calculate_all_metrics',
    'create_comparison_panel',
    'create_comparison_image',
    'create_charts_panel',
    'create_training_chart',
    'create_classification_panel',
    'format_classification_html',
    'create_upload_panel'
]
