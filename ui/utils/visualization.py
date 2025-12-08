"""
UI-Specific Visualization Utilities
Plotly-based interactive charts for Gradio dashboard
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List


def create_comparison_plot(original: np.ndarray, enhanced: np.ndarray) -> go.Figure:
    """Create interactive comparison plot"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original (Hazy)", "Enhanced")
    )
    
    fig.add_trace(
        go.Image(z=original),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Image(z=enhanced),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig
