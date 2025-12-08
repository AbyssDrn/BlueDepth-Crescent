"""
Upload Panel Component
Handle image and video uploads
"""

import gradio as gr
import numpy as np
from typing import Optional, Tuple


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate uploaded image
    
    Args:
        image: Uploaded image array
    
    Returns:
        (is_valid, message)
    """
    
    if image is None:
        return False, "No image uploaded"
    
    # Check dimensions
    if image.ndim not in [2, 3]:
        return False, "Invalid image dimensions"
    
    # Check size
    height, width = image.shape[:2]
    
    if height < 64 or width < 64:
        return False, "Image too small (minimum 64x64)"
    
    if height > 4096 or width > 4096:
        return False, "Image too large (maximum 4096x4096)"
    
    # Check channels
    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        return False, "Invalid number of channels"
    
    return True, "Image valid"


def create_upload_panel():
    """Create Gradio upload panel"""
    
    with gr.Column():
        gr.Markdown("""
        ### Upload Image or Video
        
        Supported formats:
        - **Images**: JPG, PNG, BMP, TIFF (up to 4096x4096)
        - **Videos**: MP4, AVI, MOV (coming soon)
        """)
        
        # Image upload
        image_upload = gr.Image(
            label="Upload Underwater Image",
            type="numpy",
            sources=["upload", "clipboard"],
            elem_classes="upload-zone"
        )
        
        # Video upload (placeholder for future)
        video_upload = gr.Video(
            label="Upload Underwater Video (Coming Soon)",
            visible=False
        )
        
        # Upload info
        upload_info = gr.Markdown(
            "No file uploaded yet. Drag and drop or click to upload.",
            elem_id="upload-info"
        )
        
        # Validation feedback
        validation_status = gr.Textbox(
            label="Upload Status",
            interactive=False,
            visible=False
        )
        
        def on_image_upload(image):
            if image is None:
                return "No file uploaded yet.", None
            
            is_valid, message = validate_image(image)
            
            if is_valid:
                height, width = image.shape[:2]
                size_mb = image.nbytes / (1024 * 1024)
                
                info = f"""
                 Image uploaded successfully!
                - **Resolution**: {width} x {height} pixels
                - **Size**: {size_mb:.2f} MB
                - **Format**: RGB
                """
                return info, None
            else:
                return f" Upload failed: {message}", None
        
        image_upload.change(
            fn=on_image_upload,
            inputs=image_upload,
            outputs=[upload_info, validation_status]
        )
        
        return {
            'image': image_upload,
            'video': video_upload,
            'info': upload_info,
            'status': validation_status
        }
