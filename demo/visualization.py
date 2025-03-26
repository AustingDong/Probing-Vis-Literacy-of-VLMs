import cv2
import numpy as np
import types
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import spaces
from demo.modify_llama import *
from demo.modified_attn import ModifiedLlamaAttention

class Visualization:
    def __init__(self, model, register=True):
        self.model = model
        self.gradients = []
        self.activations = []
        self.hooks = []
        if register:
            self._register_hooks()

    def _register_hooks(self):
        for layer in self.target_layers:
            self.hooks.append(layer.register_forward_hook(self._forward_hook))
            self.hooks.append(layer.register_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        print("forward_hook: self_attn_input: ", input)
        self.activations.append(output)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])

    def _modify_layers(self):
        for layer in self.target_layers:
            setattr(layer, "attn_gradients", None)
            setattr(layer, "attention_map", None)

            layer.save_attn_gradients = types.MethodType(save_attn_gradients, layer)
            layer.get_attn_gradients = types.MethodType(get_attn_gradients, layer)
            layer.save_attn_map = types.MethodType(save_attn_map, layer)
            layer.get_attn_map = types.MethodType(get_attn_map, layer)

    def _forward_activate_hooks(self, module, input, output):
        print("forward_activate_hool: module: ", module)
        print("forward_activate_hook: self_attn_input: ", input)

        attn_output, attn_weights = output  # Unpack outputs
        print("attn_output shape:", attn_output.shape)
        print("attn_weights shape:", attn_weights.shape)
        module.save_attn_map(attn_weights)
        attn_weights.register_hook(module.save_attn_gradients)

    def _register_hooks_activations(self):
        for layer in self.target_layers:
            if hasattr(layer, "q_proj"): # is an attention layer
                self.hooks.append(layer.register_forward_hook(self._forward_activate_hooks))

    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()  

    def setup_grads(self):
        torch.autograd.set_detect_anomaly(True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        for layer in self.target_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward_backward(self):
        raise NotImplementedError
    
    def grad_cam_vis(self):
        self.model.zero_grad()
        cam_sum = None
        for act, grad in zip(self.activations, self.gradients):

            act = F.relu(act[0])
            
            grad_weights = grad.mean(dim=-1, keepdim=True)
            
            # print("act shape", act.shape)
            # print("grad_weights shape", grad_weights.shape)
            
            # cam = (act * grad_weights).sum(dim=-1)
            cam, _ = (act * grad_weights).max(dim=-1)

            print("cam_shape: ", cam.shape)

            # Sum across all layers
            if cam_sum is None:
                cam_sum = cam
            else:
                cam_sum += cam  

        cam_sum = F.relu(cam_sum)
        return cam_sum
    


    def grad_cam_llm(self, mean_inside=False):

        cam_sum = None
        for act, grad in zip(self.activations, self.gradients):
            
            if mean_inside:
                act = act.mean(dim=1)
                grad = F.relu(grad.mean(dim=1))
                cam = act * grad
            else:
                cam = act * grad
                cam = act * grad.sum(dim=1)
            
            print(cam.shape)

            # Sum across all layers
            if cam_sum is None:
                cam_sum = cam
            else:
                cam_sum += cam  

        cam_sum = F.relu(cam_sum)
        return cam_sum
    
    def attention_map(self):
        raise NotImplementedError
    
    def attn_guided_cam(self):
        
        cams = []
        for act, grad in zip(self.activations, self.gradients):
            # print("act shape", act.shape)
            # print("grad shape", grad.shape)
            
            grad = F.relu(grad)

            # cam = grad
            cam = act * grad # shape: [1, heads, seq_len, seq_len]
            cam = cam.sum(dim=1) # shape: [1, seq_len, seq_len]
            cam = cam.to(torch.float32).detach().cpu()
            cams.append(cam)
        return cams
    
    
    def process(self, cam_sum, thresholding=True, remove_cls=False, normalize=True):

        cam_sum = cam_sum.to(torch.float32)

        # thresholding
        if thresholding:
            percentile = torch.quantile(cam_sum, 0.2)  # Adjust threshold dynamically
            cam_sum[cam_sum < percentile] = 0

        # Remove CLS
        if remove_cls:
            cam_sum = cam_sum[0, 1:]

        num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
        grid_size = int(num_patches ** 0.5)
        # print(f"Detected grid size: {grid_size}x{grid_size}")
        cam_sum = cam_sum.view(grid_size, grid_size).detach()

        # Normalize
        if normalize:
            cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())

        return cam_sum, grid_size
    
    def process_multiple(self, cam_sum, start_idx, images_seq_mask, thresholding=True, normalize=True):
        cam_sum = cam_sum.to(torch.float32)
        # thresholding
        if thresholding:
            percentile = torch.quantile(cam_sum, 0.2)  # Adjust threshold dynamically
            cam_sum[cam_sum < percentile] = 0


        # cam_sum shape: [1, seq_len, seq_len]
        cam_sum_lst = []
        cam_sum_raw = cam_sum
        start = start_idx
        for i in range(start, cam_sum_raw.shape[1]):
            cam_sum = cam_sum_raw[:, i, :] # shape: [1: seq_len]
            cam_sum = cam_sum[images_seq_mask].unsqueeze(0) # shape: [1, img_seq_len]
            # print("cam_sum shape: ", cam_sum.shape)
            num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
            grid_size = int(num_patches ** 0.5)
            # print(f"Detected grid size: {grid_size}x{grid_size}")

            cam_sum = cam_sum.view(grid_size, grid_size)
            if normalize:
                cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())
            cam_sum = cam_sum.detach().to("cpu")
            cam_sum_lst.append(cam_sum)
        return cam_sum_lst, grid_size
            
    def process_multiple_acc(self, cams, start_idx, images_seq_mask, normalize=False, accumulate_method="sum"):
        cam_sum_lst = []
        for i in range(start_idx, cams[0].shape[1]):
            cam_sum = None
            for layer, cam_l in enumerate(cams):
                cam_l_i = cam_l[0, i, :] # shape: [1: seq_len]

                cam_l_i = cam_l_i[images_seq_mask].unsqueeze(0) # shape: [1, img_seq_len]

                num_patches = cam_l_i.shape[-1]  # Last dimension of CAM output
                grid_size = int(num_patches ** 0.5)
                # print(f"Detected grid size: {grid_size}x{grid_size}")

                # Fix the reshaping step dynamically
                cam_reshaped = cam_l_i.view(grid_size, grid_size)

                if normalize:
                    cam_reshaped = (cam_reshaped - cam_reshaped.min()) / (cam_reshaped.max() - cam_reshaped.min())
                if cam_sum == None:
                    cam_sum = cam_reshaped
                else:
                    if accumulate_method == "sum":
                        cam_sum += cam_reshaped
                    elif accumulate_method == "mult":
                        cam_sum *= cam_reshaped + 1

            cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())
            cam_sum_lst.append(cam_sum)
        return cam_sum_lst, grid_size
    
    def generate_cam(self, input_tensor, target_token_idx=None):
        raise NotImplementedError




class VisualizationClip(Visualization):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model)

    @spaces.GPU(duration=120)
    def forward_backward(self, input_tensor, visual_method, target_token_idx):
        output_full = self.model(**input_tensor)

        if target_token_idx is None:
            target_token_idx = torch.argmax(output_full.logits, dim=1).item()

        if visual_method == "CLS":
            output = output_full.image_embeds
        elif visual_method == "avg":
            output = self.model.visual_projection(output_full.vision_model_output.last_hidden_state).mean(dim=1)
        else:
            output, _ = self.model.visual_projection(output_full.vision_model_output.last_hidden_state).max(dim=1)


        output.backward(output_full.text_embeds[target_token_idx:target_token_idx+1], retain_graph=True)
        return output_full
        
        
    @spaces.GPU(duration=120)
    def generate_cam(self, input_tensor, target_token_idx=None, visual_method="CLS"):
        """ Generates Grad-CAM heatmap for ViT. """
        self.setup_grads()
        # Forward Backward pass
        output_full = self.forward_backward(input_tensor, visual_method, target_token_idx)

        cam_sum = self.grad_cam_vis()
        cam_sum, grid_size = self.process(cam_sum)
        
        return cam_sum, output_full, grid_size
























class VisualizationJanus(Visualization):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model)
        self._modify_layers()
        self._register_hooks_activations()

    def forward_backward(self, input_tensor, tokenizer, temperature, top_p, target_token_idx=None, visual_method="softmax", focus="Visual Encoder"):
        # Forward
        image_embeddings, inputs_embeddings, outputs = self.model(input_tensor, tokenizer, temperature, top_p)
        input_ids = input_tensor.input_ids
        start_idx = 620
        self.model.zero_grad()
        if focus == "Visual Encoder":
            loss = outputs.logits.max(dim=-1).values[0, start_idx + target_token_idx]
            loss.backward()
        
        elif focus == "Language Model":
            if target_token_idx == -1:
                loss = outputs.logits.max(dim=-1).values.sum()
            else:
                loss = outputs.logits.max(dim=-1).values[0, start_idx + target_token_idx]
            loss.backward()
            
            self.activations = self.activations = [layer.attn_sigmoid_weights for layer in self.target_layers] if visual_method == "sigmoid" else [layer.get_attn_map() for layer in self.target_layers]
            self.gradients = [layer.get_attn_gradients() for layer in self.target_layers]
    
    @spaces.GPU(duration=120)
    def generate_cam(self, input_tensor, tokenizer, temperature, top_p, target_token_idx=None, visual_method="softmax", focus="Visual Encoder", accumulate_method="sum"):
        
        self.setup_grads()

        # Forward Backward pass
        self.forward_backward(input_tensor, tokenizer, temperature, top_p, target_token_idx, visual_method, focus)
        
        start_idx = 620
        if focus == "Visual Encoder":

            cam_sum = self.grad_cam_vis()
            cam_sum, grid_size = self.process(cam_sum)
            return cam_sum, grid_size, start_idx

        elif focus == "Language Model":
            
            cam_sum = self.grad_cam_llm(mean_inside=True)
            
            images_seq_mask = input_tensor.images_seq_mask
            
            cam_sum_lst, grid_size = self.process_multiple(cam_sum, start_idx, images_seq_mask)

            return cam_sum_lst, grid_size, start_idx









class VisualizationLLaVA(Visualization):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model, register=False)
        self._modify_layers()
        self._register_hooks_activations()

    def forward_backward(self, inputs):
        # Forward pass
        outputs_raw = self.model(**inputs)

        self.model.zero_grad()
        print("outputs_raw", outputs_raw)

        loss = outputs_raw.logits.max(dim=-1).values.sum()
        loss.backward()
        self.activations = [layer.get_attn_map() for layer in self.target_layers]
        self.gradients = [layer.get_attn_gradients() for layer in self.target_layers]

    @spaces.GPU(duration=120)
    def generate_cam(self, inputs, tokenizer, temperature, top_p, target_token_idx=None, visual_method="softmax", focus="Visual Encoder", accumulate_method="sum"):
        
        self.setup_grads()
        self.forward_backward(inputs)

        # get image masks
        images_seq_mask = []
        last = 0
        for i in range(inputs["input_ids"].shape[1]):
            decoded_token = tokenizer.decode(inputs["input_ids"][0][i].item())
            if (decoded_token == "<image>"):
                images_seq_mask.append(True)
                last = i
            else:
                images_seq_mask.append(False)


        # Aggregate activations and gradients from ALL layers
        start_idx = last + 1
        cams = self.attn_guided_cam()
        cam_sum_lst, grid_size = self.process_multiple_acc(cams, start_idx, images_seq_mask, accumulate_method=accumulate_method)

        return cam_sum_lst, grid_size, start_idx






class VisualizationChartGemma(Visualization):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model, register=True)
        self._modify_layers()
        self._register_hooks_activations()

    # def custom_loss(self, start_idx, input_ids, logits):
    #     Q = logits.shape[1]
    #     loss = 0
    #     q = 0
    #     while start_idx + q < Q - 1:
    #         loss += F.cross_entropy(logits[0, start_idx + q], input_ids[0, start_idx + q + 1])
    #         q += 1
    #     return loss

    
    def forward_backward(self, inputs, focus, start_idx, target_token_idx, visual_method="softmax"):
        outputs_raw = self.model(**inputs, output_hidden_states=True)
        if focus == "Visual Encoder":
            
            self.model.zero_grad()

            loss = outputs_raw.logits.max(dim=-1).values[0, start_idx + target_token_idx]
            loss.backward()
        
        elif focus == "Language Model":
            self.model.zero_grad()
            print("logits shape:", outputs_raw.logits.shape)
            if target_token_idx == -1:
                loss = outputs_raw.logits.max(dim=-1).values.sum()
                # loss = self.custom_loss(start_idx, inputs['input_ids'], outputs_raw.logits)
            else:
                loss = outputs_raw.logits.max(dim=-1).values[0, start_idx + target_token_idx]
            loss.backward()
            self.activations = [layer.attn_sigmoid_weights for layer in self.target_layers] if visual_method == "sigmoid" else [layer.get_attn_map() for layer in self.target_layers]
            self.gradients = [layer.get_attn_gradients() for layer in self.target_layers]
    
    @spaces.GPU(duration=120)
    def generate_cam(self, inputs, tokenizer, temperature, top_p, target_token_idx=None, visual_method="softmax", focus="Visual Encoder", accumulate_method="sum"):
        
        # Forward pass
        self.setup_grads()
        
        # get image masks
        images_seq_mask = []
        last = 0
        for i in range(inputs["input_ids"].shape[1]):
            decoded_token = tokenizer.decode(inputs["input_ids"][0][i].item())
            if (decoded_token == "<image>"):
                images_seq_mask.append(True)
                last = i
            else:
                images_seq_mask.append(False)
        start_idx = last + 1


        self.forward_backward(inputs, focus, start_idx, target_token_idx, visual_method)
        if focus == "Visual Encoder":
            
            cam_sum = self.grad_cam_vis()
            cam_sum, grid_size = self.process(cam_sum, remove_cls=False)

            return cam_sum, grid_size, start_idx

        elif focus == "Language Model":

            cams = self.attn_guided_cam()
            cam_sum_lst, grid_size = self.process_multiple_acc(cams, start_idx, images_seq_mask, accumulate_method=accumulate_method)

            # cams shape: [layers, 1, seq_len, seq_len]
            


        return cam_sum_lst, grid_size, start_idx










def generate_gradcam(
    cam, 
    image,
    size = (384, 384),
    alpha=0.5, 
    colormap=cv2.COLORMAP_JET, 
    aggregation='mean',
    normalize=False
):
    """
    Generates a Grad-CAM heatmap overlay on top of the input image.

    Parameters:
        cam (torch.Tensor): A tensor of shape (C, H, W) representing the
            intermediate activations or gradients at the target layer.
        image (PIL.Image): The original image.
        size (tuple): The desired size of the heatmap overlay (default (384, 384)).
        alpha (float): The blending factor for the heatmap overlay (default 0.5).
        colormap (int): OpenCV colormap to apply (default cv2.COLORMAP_JET).
        aggregation (str): How to aggregate across channels; either 'mean' or 'sum'.
        normalize (bool): Whether to normalize the heatmap (default False).

    Returns:
        PIL.Image: The image overlaid with the Grad-CAM heatmap.
    """
    # print("Generating Grad-CAM with shape:", cam.shape)

    if normalize:
        cam_min, cam_max = cam.min(), cam.max()
        cam = cam - cam_min
        cam = cam / (cam_max - cam_min)
    # Convert tensor to numpy array
    cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=size, mode='bilinear').squeeze()
    cam_np = cam.squeeze().detach().cpu().numpy()

    # Apply Gaussian blur for smoother heatmaps
    cam_np = cv2.GaussianBlur(cam_np, (5,5), sigmaX=0.8)

    # Resize the cam to match the image size
    width, height = size
    cam_resized = cv2.resize(cam_np, (width, height))

    # Convert the normalized map to a heatmap (0-255 uint8)
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    # OpenCV produces heatmaps in BGR, so convert to RGB for consistency
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original image to a numpy array
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (width, height))

    # Blend the heatmap with the original image
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    return Image.fromarray(overlay)

