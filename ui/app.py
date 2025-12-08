#!/usr/bin/env python3
"""
BlueDepth-Crescent Professional Dashboard
Production-ready Gradio interface for underwater image enhancement

Run: python ui/app.py
"""

import gradio as gr
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import json
import plotly.graph_objects as go
from typing import Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNetLight, UNetStandard, UNetAttention, UnderwaterClassifier
from inference import ImageEnhancer, UnderwaterClassifierInference
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import create_comparison_plot

# Import UI components
from ui.components import (
    create_header,
    ModelSelector,
    create_metrics_display,
    create_training_charts
)
from ui.utils.model_loader import get_best_models
from ui.utils.inference_handler import InferenceHandler


# Global inference handler
inference_handler = None


def initialize_app():
    """Initialize application and load models"""
    global inference_handler
    
    print("Initializing BlueDepth-Crescent Dashboard...")
    
    # Load best models
    models_info = get_best_models()
    
    # Initialize inference handler
    inference_handler = InferenceHandler(models_info)
    
    print(f" Loaded {len(models_info)} trained models")
    print(" Dashboard ready!")
    
    return models_info


def process_image(
    image: np.ndarray,
    model_name: str,
    enable_classification: bool = True
) -> Tuple[np.ndarray, dict, dict]:
    """
    Process uploaded image
    
    Returns:
        enhanced_image: Enhanced image array
        metrics: Enhancement metrics dict
        classification: Classification results dict
    """
    
    if image is None:
        return None, {}, {}
    
    # Enhance image
    enhanced, inference_time = inference_handler.enhance(image, model_name)
    
    # Calculate metrics
    metrics = {
        'inference_time_ms': inference_time * 1000,
        'psnr': calculate_psnr(image, enhanced) if image.shape == enhanced.shape else 0,
        'ssim': calculate_ssim(image, enhanced) if image.shape == enhanced.shape else 0,
        'resolution': f"{enhanced.shape[1]}x{enhanced.shape[0]}",
        'model': model_name
    }
    
    # Classification (if enabled)
    classification = {}
    if enable_classification:
        classification = inference_handler.classify(enhanced)
    
    return enhanced, metrics, classification


def create_comparison_view(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison"""
    if original is None or enhanced is None:
        return None
    
    # Resize to match if needed
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # Create side-by-side
    comparison = np.hstack([original, enhanced])
    
    # Add labels
    comparison = comparison.copy()
    cv2.putText(comparison, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Enhanced", (original.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return comparison


def format_metrics_display(metrics: dict) -> str:
    """Format metrics for display"""
    if not metrics:
        return "No metrics available"
    
    html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white;">
        <h3 style="margin-top: 0;">Enhancement Metrics</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
    """
    
    metric_items = [
        ("Inference Time", f"{metrics.get('inference_time_ms', 0):.2f} ms"),
        ("PSNR", f"{metrics.get('psnr', 0):.2f} dB"),
        ("SSIM", f"{metrics.get('ssim', 0):.4f}"),
        ("Resolution", metrics.get('resolution', 'N/A')),
        ("Model", metrics.get('model', 'Unknown')),
        ("FPS", f"{1000/metrics.get('inference_time_ms', 1):.1f}")
    ]
    
    for label, value in metric_items:
        html += f"""
        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <div style="font-size: 0.9em; opacity: 0.9;">{label}</div>
            <div style="font-size: 1.3em; font-weight: bold;">{value}</div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html


def format_classification_display(classification: dict) -> str:
    """Format classification results"""
    if not classification or 'detections' not in classification:
        return "<div style='padding: 20px; text-align: center;'>No objects detected</div>"
    
    html = """
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h3 style="margin-top: 0; color: #333;">Detected Objects</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: #667eea; color: white;">
                    <th style="padding: 10px; text-align: left;">Object</th>
                    <th style="padding: 10px; text-align: center;">Confidence</th>
                    <th style="padding: 10px; text-align: center;">Threat Level</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for det in classification['detections'][:10]:  # Top 10
        threat_color = {
            'high': '#dc3545',
            'medium': '#ffc107',
            'low': '#28a745',
            'none': '#6c757d'
        }.get(det.get('threat', 'none'), '#6c757d')
        
        conf_pct = det.get('confidence', 0) * 100
        
        html += f"""
        <tr style="border-bottom: 1px solid #dee2e6;">
            <td style="padding: 10px;">{det.get('class', 'Unknown')}</td>
            <td style="padding: 10px; text-align: center;">
                <span style="background: #667eea; color: white; padding: 3px 10px; border-radius: 15px;">
                    {conf_pct:.1f}%
                </span>
            </td>
            <td style="padding: 10px; text-align: center;">
                <span style="background: {threat_color}; color: white; padding: 3px 10px; border-radius: 15px;">
                    {det.get('threat', 'None').upper()}
                </span>
            </td>
        </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html


def create_training_chart(model_name: str) -> go.Figure:
    """Create interactive training metrics chart"""
    
    # Load training history
    checkpoint_path = Path(f"checkpoints/{model_name}_best.pth")
    
    if not checkpoint_path.exists():
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="Training history not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('training_history', {})
        
        if not history:
            fig = go.Figure()
            fig.add_annotation(
                text="No training history in checkpoint",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        epochs = list(range(1, len(history.get('train_loss', [])) + 1))
        
        # Training loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history.get('train_loss', []),
            name='Training Loss',
            line=dict(color='#dc3545', width=2),
            mode='lines+markers'
        ))
        
        # Validation loss
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history['val_loss'],
                name='Validation Loss',
                line=dict(color='#ffc107', width=2),
                mode='lines+markers'
            ))
        
        # PSNR (if available)
        if 'psnr' in history:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history['psnr'],
                name='PSNR',
                line=dict(color='#28a745', width=2),
                mode='lines+markers',
                yaxis='y2'
            ))
        
        fig.update_layout(
            title=f"Training Metrics - {model_name}",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="PSNR (dB)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading training history: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_dashboard():
    """Create main Gradio dashboard"""
    
    # Initialize
    models_info = initialize_app()
    model_names = list(models_info.keys())
    
    # Custom CSS with logo styling
    custom_css = """
    <style>
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1400px !important;
            margin: auto;
        }
        
        .logo-header {
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .logo-clickable {
            cursor: pointer;
            transition: transform 0.3s ease, filter 0.3s ease;
        }
        
        .logo-clickable:hover {
            transform: scale(1.08);
            filter: brightness(1.1);
        }
        
        .upload-box {
            border: 2px dashed #667eea !important;
            border-radius: 10px !important;
            padding: 20px !important;
            transition: all 0.3s ease;
        }
        
        .upload-box:hover {
            border-color: #764ba2 !important;
            background: rgba(102, 126, 234, 0.05) !important;
        }
        
        .badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            margin: 0 8px;
            font-size: 0.9em;
            font-weight: 500;
            transition: transform 0.2s ease;
        }
        
        .badge:hover {
            transform: translateY(-2px);
        }
    </style>
    """
    
    # Create Gradio interface
    with gr.Blocks(css=custom_css, title="BlueDepth-Crescent", theme=gr.themes.Soft()) as demo:
        
        # ============================================================
        # PROFESSIONAL HEADER WITH LOGO (TOP-LEFT OPTIMAL PLACEMENT)
        # ============================================================
        gr.HTML("""
        <div class="logo-header" style="
            display: flex; 
            align-items: center; 
            justify-content: space-between;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px 40px; 
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            
            <!-- LEFT SECTION: Logo + Title -->
            <div style="display: flex; align-items: center; gap: 25px;">
                <!-- LOGO (TOP-LEFT - OPTIMAL PLACEMENT) -->
                <img src="/file/ui/assets/logo.png" 
                     alt="BlueDepth-Crescent Logo" 
                     class="logo-clickable"
                     style="height: 70px; width: auto; 
                            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));"
                     onclick="window.location.reload()"
                     title="Click to refresh dashboard">
                
                <!-- TITLE + SUBTITLE -->
                <div style="color: white;">
                    <h1 style="margin: 0; font-size: 2.3em; font-weight: 700; 
                               text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        BlueDepth-Crescent
                    </h1>
                    <p style="margin: 6px 0 0 0; font-size: 1.05em; opacity: 0.95; 
                              font-weight: 300;">
                        Underwater Vision Intelligence System
                    </p>
                </div>
            </div>
            
            <!-- RIGHT SECTION: Status + Version -->
            <div style="text-align: right; color: white;">
                <div style="background: rgba(255,255,255,0.2); 
                           padding: 8px 20px; border-radius: 25px; 
                           margin-bottom: 8px;
                           backdrop-filter: blur(10px);
                           border: 1px solid rgba(255,255,255,0.3);">
                    <span style="font-size: 0.95em; font-weight: 500;">
                        🟢 System Online
                    </span>
                </div>
                <div style="font-size: 0.8em; opacity: 0.85;">
                    v1.0.0 | RTX 4050 Ready | CUDA Enabled
                </div>
            </div>
        </div>
        
        <!-- FEATURE BADGES (Under Header) -->
        <div style="text-align: center; margin-bottom: 30px;">
            <span class="badge" style="background: #667eea; color: white;">
                 PyTorch
            </span>
            <span class="badge" style="background: #28a745; color: white;">
                 CUDA Accelerated
            </span>
            <span class="badge" style="background: #17a2b8; color: white;">
                 Edge Deployment
            </span>
            <span class="badge" style="background: #ffc107; color: #333;">
                 Underwater Optimized
            </span>
        </div>
        """)
        
        # ============================================================
        # MODEL SELECTION SIDEBAR
        # ============================================================
        with gr.Row():
            with gr.Column(scale=1):
                # Model Selection
                gr.Markdown("###  Model Selection")
                model_dropdown = gr.Dropdown(
                    choices=model_names,
                    value=model_names[0] if model_names else None,
                    label="Select Enhancement Model",
                    info="Choose the best-performing model for your use case"
                )
                
                # Model info
                model_info = gr.HTML()
                
                def update_model_info(model_name):
                    if not model_name:
                        return ""
                    
                    info = models_info.get(model_name, {})
                    return f"""
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                padding: 20px; border-radius: 10px; margin-top: 10px;
                                border-left: 4px solid #667eea;">
                        <h4 style="margin-top: 0; color: #667eea;">Model Information</h4>
                        <div style="margin: 8px 0;">
                            <strong>Name:</strong> {model_name}
                        </div>
                        <div style="margin: 8px 0;">
                            <strong>Architecture:</strong> {info.get('architecture', 'Unknown')}
                        </div>
                        <div style="margin: 8px 0;">
                            <strong>Parameters:</strong> {info.get('parameters', 'N/A')}
                        </div>
                        <div style="margin: 8px 0;">
                            <strong>Best PSNR:</strong> <span style="color: #28a745; font-weight: bold;">{info.get('best_psnr', 0):.2f} dB</span>
                        </div>
                        <div style="margin: 8px 0;">
                            <strong>Training Epochs:</strong> {info.get('epochs', 'N/A')}
                        </div>
                    </div>
                    """
                
                model_dropdown.change(update_model_info, model_dropdown, model_info)
                
                # Options
                gr.Markdown("###  Options")
                enable_classification = gr.Checkbox(
                    label="Enable Object Classification",
                    value=True,
                    info="Classify and detect objects in enhanced images"
                )
        
        # ============================================================
        # UPLOAD & PROCESS SECTION
        # ============================================================
        gr.Markdown("##  Upload & Process")
        
        with gr.Row():
            # Upload panel
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Underwater Image",
                    type="numpy",
                    elem_classes="upload-box"
                )
                
                # Or upload video
                input_video = gr.Video(
                    label="Or Upload Video (Coming Soon)",
                    visible=False  # TODO: Implement video processing
                )
                
                process_btn = gr.Button(
                    " Enhance Image",
                    variant="primary",
                    size="lg"
                )
        
        # ============================================================
        # RESULTS SECTION
        # ============================================================
        gr.Markdown("##  Results")
        
        with gr.Row():
            # Original image
            with gr.Column(scale=1):
                gr.Markdown("###  Original Image (Hazy)")
                original_display = gr.Image(
                    label="Original",
                    interactive=False
                )
            
            # Enhanced image
            with gr.Column(scale=1):
                gr.Markdown("###  Enhanced Image")
                enhanced_display = gr.Image(
                    label="Enhanced",
                    interactive=False
                )
        
        # ============================================================
        # METRICS AND CLASSIFICATION
        # ============================================================
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("###  Enhancement Metrics")
                metrics_display = gr.HTML()
            
            with gr.Column(scale=1):
                gr.Markdown("###  Object Classification")
                classification_display = gr.HTML()
        
        # ============================================================
        # TRAINING ANALYTICS
        # ============================================================
        gr.Markdown("##  Training Analytics")
        
        with gr.Row():
            training_chart = gr.Plot(label="Training Metrics")
        
        # ============================================================
        # PROCESSING LOGIC
        # ============================================================
        def process_and_display(image, model_name, enable_class):
            if image is None:
                return None, None, "", "", None
            
            # Process
            enhanced, metrics, classification = process_image(
                image, model_name, enable_class
            )
            
            # Format displays
            metrics_html = format_metrics_display(metrics)
            class_html = format_classification_display(classification) if enable_class else ""
            
            # Create training chart
            chart = create_training_chart(model_name)
            
            return image, enhanced, metrics_html, class_html, chart
        
        # Connect components
        process_btn.click(
            fn=process_and_display,
            inputs=[input_image, model_dropdown, enable_classification],
            outputs=[
                original_display,
                enhanced_display,
                metrics_display,
                classification_display,
                training_chart
            ]
        )
        
        # Auto-load model info on start
        demo.load(
            fn=update_model_info,
            inputs=model_dropdown,
            outputs=model_info
        )
        
        # ============================================================
        # FOOTER
        # ============================================================
        gr.HTML("""
        <div style="text-align: center; margin-top: 50px; padding: 25px; 
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    border-radius: 10px;
                    border-top: 3px solid #667eea;">
            <p style="margin: 0; color: #495057; font-size: 0.95em;">
                <strong>BlueDepth-Crescent</strong> v1.0.0 | 
                Powered by <strong>PyTorch</strong> & <strong>Gradio</strong> | 
                © 2025 BlueDepth Team
            </p>
            <p style="margin: 8px 0 0 0; color: #6c757d; font-size: 0.85em;">
                Advanced Underwater Image Enhancement • Maritime Security • AUV/ROV Vision Systems
            </p>
        </div>
        """)
    
    return demo


def main():
    """Main entry point"""
    
    # Create dashboard
    demo = create_dashboard()
    
    # Launch
    print("\n" + "="*60)
    print(" BlueDepth-Crescent Dashboard Starting...")
    print("="*60)
    print(" Server: http://localhost:7860")
    print(" Platform: Gradio")
    print(" Status: Production Ready")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True,
        quiet=False,
        favicon_path="ui/assets/logo.png"  # Use logo as favicon
    )


if __name__ == "__main__":
    main()
