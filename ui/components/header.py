"""Professional header component"""

def create_header() -> str:
    """Create dashboard header HTML"""
    
    return """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 40px; border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 30px;">
        
        <div style="text-align: center;">
            <h1 style="margin: 0; font-size: 3em; font-weight: 700; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                BlueDepth-Crescent
            </h1>
            
            <p style="margin: 15px 0 5px 0; font-size: 1.4em; font-weight: 300;">
                Underwater Vision Intelligence System
            </p>
            
            <p style="margin: 5px 0 0 0; font-size: 1em; opacity: 0.9;">
                Advanced AI-Powered Image Enhancement & Object Classification
            </p>
            
            <div style="margin-top: 20px; padding-top: 20px; 
                        border-top: 1px solid rgba(255,255,255,0.3);">
                <span style="margin: 0 15px;"> PyTorch</span>
                <span style="margin: 0 15px;"> CUDA Accelerated</span>
                <span style="margin: 0 15px;"> Edge-Ready</span>
            </div>
        </div>
    </div>
    """
