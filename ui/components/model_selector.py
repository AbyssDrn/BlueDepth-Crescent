"""Model selection component"""

class ModelSelector:
    """Handle model selection logic"""
    
    def __init__(self, models_info: dict):
        self.models_info = models_info
    
    def get_model_choices(self) -> list:
        """Get list of available models"""
        return list(self.models_info.keys())
    
    def get_model_info(self, model_name: str) -> dict:
        """Get information about specific model"""
        return self.models_info.get(model_name, {})
    
    def format_model_card(self, model_name: str) -> str:
        """Format model information card"""
        
        info = self.get_model_info(model_name)
        
        return f"""
        <div style="background: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #667eea;">{model_name}</h3>
            
            <div style="margin: 10px 0;">
                <strong>Architecture:</strong> {info.get('architecture', 'Unknown')}
            </div>
            
            <div style="margin: 10px 0;">
                <strong>Parameters:</strong> {info.get('parameters', 'N/A')}
            </div>
            
            <div style="margin: 10px 0;">
                <strong>Best PSNR:</strong> {info.get('best_psnr', 0):.2f} dB
            </div>
            
            <div style="margin: 10px 0;">
                <strong>Training Epochs:</strong> {info.get('epochs', 'N/A')}
            </div>
            
            <div style="margin-top: 15px; padding-top: 15px; 
                        border-top: 1px solid #dee2e6;">
                <small style="color: #6c757d;">
                    Trained on underwater dataset | Ready for deployment
                </small>
            </div>
        </div>
        """
