"""
Training Charts Panel
Interactive Plotly charts for training metrics
"""

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def create_training_chart(
    model_name: str,
    checkpoint_dir: str = "checkpoints"
) -> go.Figure:
    """
    Create comprehensive training metrics chart
    
    Args:
        model_name: Name of the model
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Plotly figure
    """
    
    checkpoint_path = Path(checkpoint_dir) / f"{model_name}_best.pth"
    
    if not checkpoint_path.exists():
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Training history not available for {model_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=18, color="#6c757d")
        )
        fig.update_layout(
            height=500,
            template="plotly_white",
            paper_bgcolor='#f8f9fa'
        )
        return fig
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('training_history', {})
        
        if not history or 'train_loss' not in history:
            fig = go.Figure()
            fig.add_annotation(
                text="No training history in checkpoint",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Training & Validation Loss',
                'PSNR Over Time',
                'SSIM Over Time',
                'Learning Rate Schedule'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Loss curves
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history['train_loss'],
                name='Train Loss',
                line=dict(color='#dc3545', width=2),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['val_loss'],
                    name='Val Loss',
                    line=dict(color='#ffc107', width=2),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            # Mark best epoch
            best_epoch = np.argmin(history['val_loss']) + 1
            best_loss = min(history['val_loss'])
            fig.add_trace(
                go.Scatter(
                    x=[best_epoch],
                    y=[best_loss],
                    name='Best Epoch',
                    mode='markers',
                    marker=dict(size=15, color='#28a745', symbol='star')
                ),
                row=1, col=1
            )
        
        # PSNR
        if 'psnr' in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['psnr'],
                    name='PSNR',
                    line=dict(color='#28a745', width=2),
                    mode='lines+markers',
                    fill='tozeroy',
                    fillcolor='rgba(40, 167, 69, 0.1)'
                ),
                row=1, col=2
            )
        
        # SSIM
        if 'ssim' in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['ssim'],
                    name='SSIM',
                    line=dict(color='#17a2b8', width=2),
                    mode='lines+markers',
                    fill='tozeroy',
                    fillcolor='rgba(23, 162, 184, 0.1)'
                ),
                row=2, col=1
            )
        
        # Learning rate
        if 'learning_rate' in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['learning_rate'],
                    name='Learning Rate',
                    line=dict(color='#6f42c1', width=2),
                    mode='lines'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2)
        fig.update_yaxes(title_text="SSIM", row=2, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=2)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            template="plotly_white",
            hovermode='x unified',
            title=dict(
                text=f"Training Analytics - {model_name}",
                font=dict(size=20, color='#333'),
                x=0.5,
                xanchor='center'
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading training history: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#dc3545')
        )
        return fig


def create_metrics_comparison_chart(
    models_info: Dict[str, dict]
) -> go.Figure:
    """
    Create bar chart comparing model metrics
    
    Args:
        models_info: Dictionary of model information
    
    Returns:
        Plotly figure
    """
    
    model_names = list(models_info.keys())
    psnr_values = [info.get('best_psnr', 0) for info in models_info.values()]
    params = [info.get('parameters', '0') for info in models_info.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=model_names,
        y=psnr_values,
        text=[f"{p:.2f} dB" for p in psnr_values],
        textposition='auto',
        marker=dict(
            color=psnr_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="PSNR (dB)")
        ),
        name='Best PSNR'
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="PSNR (dB)",
        height=400,
        template="plotly_white"
    )
    
    return fig


def create_charts_panel():
    """Create Gradio charts panel"""
    
    with gr.Column():
        gr.Markdown("## Training Analytics")
        
        training_chart = gr.Plot(
            label="Training Metrics",
            elem_id="training-chart"
        )
        
        gr.Markdown("## Model Comparison")
        
        comparison_chart = gr.Plot(
            label="Performance Comparison",
            elem_id="comparison-chart"
        )
        
        return {
            'training': training_chart,
            'comparison': comparison_chart
        }
