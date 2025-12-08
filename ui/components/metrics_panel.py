"""
Metrics Display Panel
Show comprehensive underwater image quality metrics
"""

import gradio as gr
import numpy as np
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_uiqm,
    calculate_mae,
    calculate_mse
)


def calculate_uciqe(img: np.ndarray) -> float:
    """
    Calculate Underwater Color Image Quality Evaluation (UCIQE)
    Reference: Yang & Sowmya (2015)
    
    Args:
        img: RGB image (0-255)
    
    Returns:
        UCIQE score (higher is better)
    """
    import cv2
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Chroma
    chroma = np.sqrt(lab[:, :, 1]**2 + lab[:, :, 2]**2)
    
    # Saturation
    saturation = chroma / np.sqrt(chroma**2 + lab[:, :, 0]**2 + 1e-10)
    
    # Contrast
    luminance = lab[:, :, 0]
    contrast = luminance.std()
    
    # UCIQE formula
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    
    uciqe = (c1 * contrast) + (c2 * saturation.mean()) + (c3 * chroma.mean())
    
    return float(uciqe)


def calculate_uism(img: np.ndarray) -> float:
    """
    Calculate Underwater Image Sharpness Measure (UISM)
    
    Args:
        img: RGB image
    
    Returns:
        UISM score (higher is better)
    """
    import cv2
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Edge magnitude
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Sharpness measure
    uism = np.mean(edges)
    
    return float(uism)


def calculate_all_metrics(
    original: np.ndarray,
    enhanced: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all available metrics
    
    Args:
        original: Original hazy image
        enhanced: Enhanced image
    
    Returns:
        Dictionary of all metrics
    """
    
    metrics = {}
    
    try:
        # Reference-based metrics (comparing enhanced to original)
        if original.shape == enhanced.shape:
            metrics['psnr'] = calculate_psnr(original, enhanced)
            metrics['ssim'] = calculate_ssim(original, enhanced)
            metrics['mae'] = calculate_mae(original, enhanced)
            metrics['mse'] = calculate_mse(original, enhanced)
    except Exception as e:
        print(f"Error calculating reference metrics: {e}")
    
    try:
        # No-reference underwater metrics (enhanced image quality)
        metrics['uiqm'] = calculate_uiqm(enhanced)
        metrics['uciqe'] = calculate_uciqe(enhanced)
        metrics['uism'] = calculate_uism(enhanced)
        
        # Original image metrics for comparison
        metrics['original_uiqm'] = calculate_uiqm(original)
        metrics['original_uciqe'] = calculate_uciqe(original)
        
        # Improvement metrics
        metrics['uiqm_improvement'] = metrics['uiqm'] - metrics['original_uiqm']
        metrics['uciqe_improvement'] = metrics['uciqe'] - metrics['original_uciqe']
        
    except Exception as e:
        print(f"Error calculating underwater metrics: {e}")
    
    return metrics


def create_metrics_html(metrics: Dict[str, float], inference_time: float = 0) -> str:
    """
    Create beautiful HTML display for metrics
    
    Args:
        metrics: Dictionary of metrics
        inference_time: Inference time in seconds
    
    Returns:
        HTML string
    """
    
    if not metrics:
        return """
        <div style="padding: 30px; text-align: center; background: #f8f9fa; border-radius: 10px;">
            <p style="color: #6c757d; font-size: 1.1em;">No metrics available yet. Upload and process an image first.</p>
        </div>
        """
    
    # Calculate FPS
    fps = 1 / inference_time if inference_time > 0 else 0
    
    html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; color: white; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        
        <h2 style="margin-top: 0; text-align: center; font-size: 1.8em;">
            Enhancement Metrics
        </h2>
        
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="margin-top: 0; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;">
                Performance Metrics
            </h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 15px;">
    """
    
    # Performance metrics
    perf_metrics = {
        'Inference Time': f"{inference_time * 1000:.2f} ms",
        'Throughput': f"{fps:.1f} FPS"
    }
    
    for label, value in perf_metrics.items():
        html += f"""
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">{label}</div>
            <div style="font-size: 1.5em; font-weight: bold;">{value}</div>
        </div>
        """
    
    html += """
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="margin-top: 0; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;">
                Underwater Quality Metrics
            </h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
    """
    
    # Underwater metrics with descriptions
    underwater_metrics = [
        ('UIQM', metrics.get('uiqm', 0), 'Underwater Image Quality Measure'),
        ('UCIQE', metrics.get('uciqe', 0), 'Underwater Color Image Quality'),
        ('UISM', metrics.get('uism', 0), 'Underwater Image Sharpness')
    ]
    
    for label, value, desc in underwater_metrics:
        html += f"""
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85em; opacity: 0.9; margin-bottom: 3px;">{label}</div>
            <div style="font-size: 1.8em; font-weight: bold; margin: 5px 0;">{value:.4f}</div>
            <div style="font-size: 0.75em; opacity: 0.8;">{desc}</div>
        </div>
        """
    
    html += """
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="margin-top: 0; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;">
                Reference Quality Metrics
            </h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
    """
    
    # Reference metrics
    ref_metrics = [
        ('PSNR', metrics.get('psnr', 0), 'dB', 'Peak Signal-to-Noise Ratio'),
        ('SSIM', metrics.get('ssim', 0), '', 'Structural Similarity'),
        ('MAE', metrics.get('mae', 0), '', 'Mean Absolute Error'),
        ('MSE', metrics.get('mse', 0), '', 'Mean Squared Error')
    ]
    
    for label, value, unit, desc in ref_metrics:
        html += f"""
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85em; opacity: 0.9; margin-bottom: 3px;">{label}</div>
            <div style="font-size: 1.5em; font-weight: bold; margin: 5px 0;">{value:.3f}</div>
            <div style="font-size: 0.8em; opacity: 0.8;">{unit if unit else desc[:15]}</div>
        </div>
        """
    
    html += """
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="margin-top: 0; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;">
                Improvement Analysis
            </h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 15px;">
    """
    
    # Improvement metrics
    uiqm_imp = metrics.get('uiqm_improvement', 0)
    uciqe_imp = metrics.get('uciqe_improvement', 0)
    
    improvements = [
        ('UIQM Improvement', uiqm_imp, uiqm_imp > 0),
        ('UCIQE Improvement', uciqe_imp, uciqe_imp > 0)
    ]
    
    for label, value, is_positive in improvements:
        color = '#28a745' if is_positive else '#dc3545'
        arrow = '↑' if is_positive else '↓'
        
        html += f"""
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">{label}</div>
            <div style="font-size: 1.8em; font-weight: bold; color: {color};">
                {arrow} {abs(value):.4f}
            </div>
        </div>
        """
    
    html += """
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 20px; padding-top: 15px; 
                    border-top: 1px solid rgba(255,255,255,0.3);">
            <p style="margin: 0; font-size: 0.9em; opacity: 0.8;">
                All metrics calculated in real-time • Higher scores indicate better quality
            </p>
        </div>
    </div>
    """
    
    return html


def create_metrics_panel():
    """Create Gradio metrics panel component"""
    
    with gr.Column():
        gr.Markdown("### Enhancement Metrics")
        
        metrics_display = gr.HTML(
            value=create_metrics_html({}, 0),
            elem_id="metrics-panel"
        )
        
        return metrics_display
