"""
Image Comparison Panel
Side-by-side original vs enhanced comparison
"""

import gradio as gr
import numpy as np
import cv2
from typing import Tuple, Optional


def create_comparison_image(
    original: np.ndarray,
    enhanced: np.ndarray,
    add_labels: bool = True
) -> np.ndarray:
    """
    Create side-by-side comparison image
    
    Args:
        original: Original image
        enhanced: Enhanced image
        add_labels: Whether to add text labels
    
    Returns:
        Combined comparison image
    """
    
    if original is None or enhanced is None:
        return None
    
    # Resize to match if needed
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # Create side-by-side
    comparison = np.hstack([original, enhanced])
    
    # Add labels
    if add_labels:
        comparison = comparison.copy()
        
        # Add "Original" label
        cv2.putText(
            comparison,
            "ORIGINAL (HAZY)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            comparison,
            "ORIGINAL (HAZY)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        
        # Add "Enhanced" label
        cv2.putText(
            comparison,
            "ENHANCED",
            (original.shape[1] + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            comparison,
            "ENHANCED",
            (original.shape[1] + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # Add divider line
        height = comparison.shape[0]
        cv2.line(
            comparison,
            (original.shape[1], 0),
            (original.shape[1], height),
            (255, 255, 255),
            3
        )
    
    return comparison


def create_slider_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    slider_position: float = 0.5
) -> np.ndarray:
    """
    Create slider-style comparison
    
    Args:
        original: Original image
        enhanced: Enhanced image
        slider_position: Position of slider (0-1)
    
    Returns:
        Blended comparison image
    """
    
    if original is None or enhanced is None:
        return None
    
    # Resize to match
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # Create mask based on slider position
    width = original.shape[1]
    split_point = int(width * slider_position)
    
    # Create comparison
    comparison = original.copy()
    comparison[:, split_point:] = enhanced[:, split_point:]
    
    # Add slider line
    cv2.line(
        comparison,
        (split_point, 0),
        (split_point, comparison.shape[0]),
        (255, 255, 0),
        3
    )
    
    return comparison


def create_comparison_panel():
    """Create Gradio comparison panel"""
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Original Image (Hazy)")
            original_display = gr.Image(
                label="Original",
                type="numpy",
                interactive=False,
                elem_id="original-image"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Enhanced Image")
            enhanced_display = gr.Image(
                label="Enhanced",
                type="numpy",
                interactive=False,
                elem_id="enhanced-image"
            )
    
    # Comparison options
    with gr.Row():
        show_sidebyside = gr.Checkbox(
            label="Show Side-by-Side Comparison",
            value=True
        )
        
        comparison_slider = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            label="Comparison Slider",
            visible=False
        )
    
    # Combined comparison view
    comparison_display = gr.Image(
        label="Comparison View",
        type="numpy",
        interactive=False,
        visible=False
    )
    
    return {
        'original': original_display,
        'enhanced': enhanced_display,
        'comparison': comparison_display,
        'slider': comparison_slider,
        'show_sidebyside': show_sidebyside
    }
