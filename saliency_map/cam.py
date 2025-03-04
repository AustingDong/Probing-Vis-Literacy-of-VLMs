import torch
import cv2
from PIL import Image
import numpy as np

class MultimodalGradCAM:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        # Hook the last vision transformer layer
        def forward_hook(module, input, output):
            self.activations['vision'] = output.last_hidden_state
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients['vision'] = grad_output[0]
            
        vision_encoder = self.model.get_vision_encoder()
        vision_encoder.layers[-1].register_forward_hook(forward_hook)
        vision_encoder.layers[-1].register_backward_hook(backward_hook)

    def generate_saliency(self, image, question):
        # Preprocess inputs
        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Forward pass
        outputs = self.model(**inputs)
        answer_ids = outputs.logits.argmax(dim=-1)
        
        # Get target token (use last token for answer)
        target_token_id = answer_ids[0, -1].item()
        target = outputs.logits[0, -1, target_token_id]
        
        # Backward pass
        self.model.zero_grad()
        target.backward()
        
        # Process activations and gradients
        activations = self.activations['vision'].detach()
        gradients = self.gradients['vision'].detach()
        
        # Grad-CAM calculation
        weights = gradients.mean(dim=[1, 2], keepdim=True)  # Global average pooling
        cam = (weights * activations).sum(dim=-1, keepdims=True)
        cam = torch.relu(cam)
        
        # Reshape and normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def visualize(self, image, cam):
        # Resize CAM to original image size
        img_size = image.size[::-1]  # (width, height) -> (height, width)
        cam = cv2.resize(cam, img_size)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose on original image
        superimposed = np.array(image) * 0.4 + heatmap * 0.6
        return Image.fromarray(np.uint8(superimposed))
