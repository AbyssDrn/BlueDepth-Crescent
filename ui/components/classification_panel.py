"""
Object Classification Display Panel
Show detected objects with confidence and threat levels
"""

import gradio as gr
from typing import Dict, List
import numpy as np
import cv2


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict]
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
    
    Returns:
        Image with bounding boxes
    """
    
    img_with_boxes = image.copy()
    
    for det in detections:
        if 'bbox' not in det:
            continue
        
        bbox = det['bbox']  # [x, y, w, h]
        confidence = det.get('confidence', 0)
        class_name = det.get('class', 'Unknown')
        threat = det.get('threat', 'none')
        
        # Color based on threat level
        threat_colors = {
            'high': (255, 0, 0),      # Red
            'medium': (255, 165, 0),  # Orange
            'low': (255, 255, 0),     # Yellow
            'none': (0, 255, 0)       # Green
        }
        color = threat_colors.get(threat, (0, 255, 0))
        
        # Draw rectangle
        x, y, w, h = bbox
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            img_with_boxes,
            (x, y - label_h - 10),
            (x + label_w, y),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_with_boxes,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return img_with_boxes


def format_classification_html(classification: Dict) -> str:
    """
    Format classification results as HTML table
    
    Args:
        classification: Classification results dictionary
    
    Returns:
        HTML string
    """
    
    if not classification or 'detections' not in classification:
        return """
        <div style="padding: 30px; text-align: center; background: #f8f9fa; border-radius: 10px;">
            <p style="color: #6c757d; font-size: 1.1em;">
                No objects detected. Enable classification and process an image.
            </p>
        </div>
        """
    
    detections = classification['detections']
    
    if not detections:
        return """
        <div style="padding: 30px; text-align: center; background: #f8f9fa; border-radius: 10px;">
            <p style="color: #6c757d; font-size: 1.1em;">No objects detected in this image.</p>
        </div>
        """
    
    html = """
    <div style="background: #ffffff; padding: 20px; border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: #333;">Detected Objects</h3>
            <span style="background: #667eea; color: white; padding: 5px 15px; 
                         border-radius: 20px; font-size: 0.9em;">
                {count} objects
            </span>
        </div>
        
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white;">
                    <th style="padding: 12px; text-align: left; border-radius: 5px 0 0 0;">Object</th>
                    <th style="padding: 12px; text-align: center;">Confidence</th>
                    <th style="padding: 12px; text-align: center;">Threat Level</th>
                    <th style="padding: 12px; text-align: center; border-radius: 0 5px 0 0;">Location</th>
                </tr>
            </thead>
            <tbody>
    """.format(count=len(detections))
    
    for idx, det in enumerate(detections[:15]):  # Top 15
        class_name = det.get('class', 'Unknown')
        confidence = det.get('confidence', 0) * 100
        threat = det.get('threat', 'none').upper()
        bbox = det.get('bbox', [0, 0, 0, 0])
        
        # Threat color
        threat_colors = {
            'HIGH': '#dc3545',
            'MEDIUM': '#ffc107',
            'LOW': '#28a745',
            'NONE': '#6c757d'
        }
        threat_color = threat_colors.get(threat, '#6c757d')
        
        # Row background (alternating)
        bg_color = '#f8f9fa' if idx % 2 == 0 else '#ffffff'
        
        html += f"""
        <tr style="background: {bg_color}; border-bottom: 1px solid #dee2e6;">
            <td style="padding: 12px; font-weight: 500;">{class_name}</td>
            <td style="padding: 12px; text-align: center;">
                <div style="background: #667eea; color: white; padding: 4px 12px; 
                            border-radius: 15px; display: inline-block; font-weight: 500;">
                    {confidence:.1f}%
                </div>
            </td>
            <td style="padding: 12px; text-align: center;">
                <span style="background: {threat_color}; color: white; padding: 4px 12px; 
                             border-radius: 15px; font-weight: 500; font-size: 0.9em;">
                    {threat}
                </span>
            </td>
            <td style="padding: 12px; text-align: center; font-family: monospace; 
                       font-size: 0.85em; color: #6c757d;">
                ({bbox[0]}, {bbox[1]})
            </td>
        </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; 
                    border-radius: 8px; text-align: center;">
            <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
                Classification powered by UnderwaterClassifier model | 
                Real-time object detection and threat assessment
            </p>
        </div>
    </div>
    """
    
    return html


def create_classification_panel():
    """Create Gradio classification panel"""
    
    with gr.Column():
        gr.Markdown("### Object Classification Results")
        
        classification_html = gr.HTML(
            value=format_classification_html({}),
            elem_id="classification-panel"
        )
        
        # Optional: Image with bounding boxes
        with gr.Accordion("Show Detection Visualization", open=False):
            detection_image = gr.Image(
                label="Detected Objects",
                type="numpy",
                interactive=False
            )
        
        return {
            'html': classification_html,
            'image': detection_image
        }
