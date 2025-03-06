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


class AttentionGuidedCAM:
    def __init__(self, model, register=True):
        self.model = model
        self.gradients = []
        self.activations = []
        self.hooks = []
        if register:
            self._register_hooks()

    def _register_hooks(self):
        """ Registers hooks to extract activations and gradients from ALL attention layers. """
        for layer in self.target_layers:
            self.hooks.append(layer.register_forward_hook(self._forward_hook))
            self.hooks.append(layer.register_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        """ Stores attention maps (before softmax) """
        self.activations.append(output)

    def _backward_hook(self, module, grad_in, grad_out):
        """ Stores gradients """
        self.gradients.append(grad_out[0])

    
    def remove_hooks(self):
        """ Remove hooks after usage. """
        for hook in self.hooks:
            hook.remove()  
    
    @spaces.GPU(duration=120)
    def generate_cam(self, input_tensor, class_idx=None):
        raise NotImplementedError




class AttentionGuidedCAMClip(AttentionGuidedCAM):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model)

    @spaces.GPU(duration=120)
    def generate_cam(self, input_tensor, class_idx=None, visual_pooling_method="CLS"):
        """ Generates Grad-CAM heatmap for ViT. """
        
        # Forward pass
        output_full = self.model(**input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output_full.logits, dim=1).item()

        if visual_pooling_method == "CLS":
            output = output_full.image_embeds
        elif visual_pooling_method == "avg":
            output = self.model.visual_projection(output_full.vision_model_output.last_hidden_state).mean(dim=1)
        else:
            # project -> pooling
            output, _ = self.model.visual_projection(output_full.vision_model_output.last_hidden_state).max(dim=1)

            # pooling -> project
            # output_mx, _ = output_full.vision_model_output.last_hidden_state.max(dim=1)
            # output = self.model.visual_projection(output_mx)

        output.backward(output_full.text_embeds[class_idx:class_idx+1], retain_graph=True)

        # Aggregate activations and gradients from ALL layers
        self.model.zero_grad()
        cam_sum = None
        for act, grad in zip(self.activations, self.gradients):

            # act = torch.sigmoid(act[0])
            act = F.relu(act[0])
            
            grad_weights = grad.mean(dim=-1, keepdim=True)
            

            print("act shape", act.shape)
            print("grad_weights shape", grad_weights.shape)
            
            # cam = (act * grad_weights).sum(dim=-1)  # Weighted activation map
            cam, _ = (act * grad_weights).max(dim=-1)
            # cam, _ = grad_weights.max(dim=-1)
            # cam = self.normalize(cam)
            print("cam_shape: ", cam.shape)

            # Sum across all layers
            if cam_sum is None:
                cam_sum = cam
            else:
                cam_sum += cam  
                

        # Normalize
        cam_sum = F.relu(cam_sum)

        # thresholding
        cam_sum = cam_sum.to(torch.float32)
        percentile = torch.quantile(cam_sum, 0.2)  # Adjust threshold dynamically
        cam_sum[cam_sum < percentile] = 0

        # Reshape
        print("cam_sum shape: ", cam_sum.shape)
        cam_sum = cam_sum[0, 1:]

        num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
        grid_size = int(num_patches ** 0.5)
        print(f"Detected grid size: {grid_size}x{grid_size}")
        
        cam_sum = cam_sum.view(grid_size, grid_size).detach()
        cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())

        return cam_sum, output_full, grid_size


class AttentionGuidedCAMJanus(AttentionGuidedCAM):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model)
        self._modify_layers()
        self._register_hooks_activations()

    def _modify_layers(self):
        for layer in self.target_layers:
            setattr(layer, "attn_gradients", None)
            setattr(layer, "attention_map", None)

            layer.save_attn_gradients = types.MethodType(save_attn_gradients, layer)
            layer.get_attn_gradients = types.MethodType(get_attn_gradients, layer)
            layer.save_attn_map = types.MethodType(save_attn_map, layer)
            layer.get_attn_map = types.MethodType(get_attn_map, layer)

    def _forward_activate_hooks(self, module, input, output):
        attn_output, attn_weights = output  # Unpack outputs
        module.save_attn_map(attn_weights)
        attn_weights.register_hook(module.save_attn_gradients)

    def _register_hooks_activations(self):
        for layer in self.target_layers:
            if hasattr(layer, "q_proj"): # is an attention layer
                self.hooks.append(layer.register_forward_hook(self._forward_activate_hooks))

    @spaces.GPU(duration=120)
    def generate_cam(self, input_tensor, tokenizer, temperature, top_p, class_idx=None, visual_pooling_method="CLS", focus="Visual Encoder"):
        """ Generates Grad-CAM heatmap for ViT. """
        
        
        # Forward pass
        image_embeddings, inputs_embeddings, outputs = self.model(input_tensor, tokenizer, temperature, top_p)


        input_ids = input_tensor.input_ids

        if focus == "Visual Encoder":
            # Pooling
            if visual_pooling_method == "CLS":
                image_embeddings_pooled = image_embeddings[:, 0, :]
            elif visual_pooling_method == "avg":
                image_embeddings_pooled = image_embeddings[:, 1:, :].mean(dim=1) # end of image: 618
            elif visual_pooling_method == "max":
                image_embeddings_pooled, _ = image_embeddings[:, 1:, :].max(dim=1)

            print("image_embeddings_shape: ", image_embeddings_pooled.shape)
            


            inputs_embeddings_pooled = inputs_embeddings[:, 620: -4].mean(dim=1)
            self.model.zero_grad()
            image_embeddings_pooled.backward(inputs_embeddings_pooled, retain_graph=True)

            cam_sum = None
            for act, grad in zip(self.activations, self.gradients):
                # act = torch.sigmoid(act)
                act = F.relu(act[0])
    

                # Compute mean of gradients
                print("grad shape:", grad.shape)
                grad_weights = grad.mean(dim=-1, keepdim=True)

                print("act shape", act.shape)
                print("grad_weights shape", grad_weights.shape)

                cam, _ = (act * grad_weights).max(dim=-1)
                print(cam.shape)

                # Sum across all layers
                if cam_sum is None:
                    cam_sum = cam
                else:
                    cam_sum += cam  

            # Normalize
            cam_sum = F.relu(cam_sum)
            

            # thresholding
            cam_sum = cam_sum.to(torch.float32)
            percentile = torch.quantile(cam_sum, 0.2)  # Adjust threshold dynamically
            cam_sum[cam_sum < percentile] = 0

            # Reshape
            # if visual_pooling_method == "CLS":
            cam_sum = cam_sum[0, 1:]
            print("cam_sum shape: ", cam_sum.shape)
            num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
            grid_size = int(num_patches ** 0.5)
            print(f"Detected grid size: {grid_size}x{grid_size}")
            
            cam_sum = cam_sum.view(grid_size, grid_size)
            cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())
            cam_sum = cam_sum.detach().to("cpu")

            return cam_sum, grid_size






        elif focus == "Language Model":
            self.model.zero_grad()
            loss = outputs.logits.max(dim=-1).values.sum()
            loss.backward()
            
            self.activations = [layer.get_attn_map() for layer in self.target_layers]
            self.gradients = [layer.get_attn_gradients() for layer in self.target_layers]

            cam_sum = None
            for act, grad in zip(self.activations, self.gradients):
                # act = torch.sigmoid(act)
                print("act_shape:", act.shape)
                # print("act1_shape:", act[1].shape)
                
                act = act.mean(dim=1)
    

                # Compute mean of gradients
                print("grad_shape:", grad.shape)
                grad_weights = F.relu(grad.mean(dim=1))


                # cam, _ = (act * grad_weights).max(dim=-1)
                # cam = act * grad_weights
                cam = act * grad_weights
                print(cam.shape)

                # Sum across all layers
                if cam_sum is None:
                    cam_sum = cam
                else:
                    cam_sum += cam  

            # Normalize
            cam_sum = F.relu(cam_sum)
            # cam_sum = cam_sum - cam_sum.min()
            # cam_sum = cam_sum / cam_sum.max()

            # thresholding
            cam_sum = cam_sum.to(torch.float32)
            percentile = torch.quantile(cam_sum, 0.2)  # Adjust threshold dynamically
            cam_sum[cam_sum < percentile] = 0

            # Reshape
            # if visual_pooling_method == "CLS":
            # cam_sum = cam_sum[0, 1:]

            # cam_sum shape: [1, seq_len, seq_len]
            cam_sum_lst = []
            cam_sum_raw = cam_sum
            for i in range(620, cam_sum_raw.shape[1]):
                cam_sum = cam_sum_raw[:, i, :] # shape: [1: seq_len]
                cam_sum = cam_sum[input_tensor.images_seq_mask].unsqueeze(0) # shape: [1, 576]
                print("cam_sum shape: ", cam_sum.shape)
                num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
                grid_size = int(num_patches ** 0.5)
                print(f"Detected grid size: {grid_size}x{grid_size}")

                # Fix the reshaping step dynamically
                
                cam_sum = cam_sum.view(grid_size, grid_size)
                cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())
                cam_sum = cam_sum.detach().to("cpu")
                cam_sum_lst.append(cam_sum)


            return cam_sum_lst, grid_size

        # Aggregate activations and gradients from ALL layers
        









class AttentionGuidedCAMLLaVA(AttentionGuidedCAM):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model, register=False)
        self._modify_layers()
        self._register_hooks_activations()

    def _modify_layers(self):
        for layer in self.target_layers:
            setattr(layer, "attn_gradients", None)
            setattr(layer, "attention_map", None)

            layer.save_attn_gradients = types.MethodType(save_attn_gradients, layer)
            layer.get_attn_gradients = types.MethodType(get_attn_gradients, layer)
            layer.save_attn_map = types.MethodType(save_attn_map, layer)
            layer.get_attn_map = types.MethodType(get_attn_map, layer)

    def _forward_activate_hooks(self, module, input, output):
        attn_output, attn_weights = output  # Unpack outputs
        attn_weights.requires_grad_()
        module.save_attn_map(attn_weights)
        attn_weights.register_hook(module.save_attn_gradients)

    def _register_hooks_activations(self):
        for layer in self.target_layers:
            if hasattr(layer, "q_proj"): # is an attention layer
                self.hooks.append(layer.register_forward_hook(self._forward_activate_hooks))

    @spaces.GPU(duration=120)
    def generate_cam(self, inputs, tokenizer, temperature, top_p, class_idx=None, visual_pooling_method="CLS", focus="Visual Encoder"):
        """ Generates Grad-CAM heatmap for ViT. """
        
        # Forward pass
        outputs_raw = self.model(**inputs)

        self.model.zero_grad()
        print("outputs_raw", outputs_raw)

        loss = outputs_raw.logits.max(dim=-1).values.sum()
        loss.backward()

        # get image masks
        image_mask = []
        last = 0
        for i in range(inputs["input_ids"].shape[1]):
            decoded_token = tokenizer.decode(inputs["input_ids"][0][i].item())
            if (decoded_token == "<image>"):
                image_mask.append(True)
                last = i
            else:
                image_mask.append(False)


        # Aggregate activations and gradients from ALL layers
        self.activations = [layer.get_attn_map() for layer in self.target_layers]
        self.gradients = [layer.get_attn_gradients() for layer in self.target_layers]
        cam_sum = None

        # Ver 2
        for act, grad in zip(self.activations, self.gradients):

            print("act shape", act.shape)
            print("grad shape", grad.shape)

            grad = F.relu(grad)


            cam = act * grad # shape: [1, heads, seq_len, seq_len]
            cam = cam.sum(dim=1) # shape: [1, seq_len, seq_len]

            # Sum across all layers
            if cam_sum is None:
                cam_sum = cam
            else:
                cam_sum += cam 

        cam_sum = F.relu(cam_sum)
        cam_sum = cam_sum.to(torch.float32)

        # thresholding
        # percentile = torch.quantile(cam_sum, 0.4)  # Adjust threshold dynamically
        # cam_sum[cam_sum < percentile] = 0

        # Reshape
        # if visual_pooling_method == "CLS":
        # cam_sum = cam_sum[0, 1:]

        # cam_sum shape: [1, seq_len, seq_len]
        cam_sum_lst = []
        cam_sum_raw = cam_sum
        start_idx = last + 1
        for i in range(start_idx, cam_sum_raw.shape[1]):
            cam_sum = cam_sum_raw[0, i, :] # shape: [1: seq_len]
            # cam_sum_min = cam_sum.min()
            # cam_sum_max = cam_sum.max()
            # cam_sum = (cam_sum - cam_sum_min) / (cam_sum_max - cam_sum_min)
            cam_sum = cam_sum[image_mask].unsqueeze(0) # shape: [1, 1024]
            print("cam_sum shape: ", cam_sum.shape)
            num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
            grid_size = int(num_patches ** 0.5)
            print(f"Detected grid size: {grid_size}x{grid_size}")

            # Fix the reshaping step dynamically
            
            cam_sum = cam_sum.view(grid_size, grid_size)
            cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())
            cam_sum_lst.append(cam_sum)


        return cam_sum_lst, grid_size



















class AttentionGuidedCAMChartGemma(AttentionGuidedCAM):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model, register=False)
        self._modify_layers()
        self._register_hooks_activations()
    
    def _modify_layers(self):
        for layer in self.target_layers:
            setattr(layer, "attn_gradients", None)
            setattr(layer, "attention_map", None)

            layer.save_attn_gradients = types.MethodType(save_attn_gradients, layer)
            layer.get_attn_gradients = types.MethodType(get_attn_gradients, layer)
            layer.save_attn_map = types.MethodType(save_attn_map, layer)
            layer.get_attn_map = types.MethodType(get_attn_map, layer)

    def _forward_activate_hooks(self, module, input, output):
        attn_output, attn_weights = output  # Unpack outputs
        print("attn_output shape:", attn_output.shape)
        print("attn_weights shape:", attn_weights.shape)
        module.save_attn_map(attn_weights)
        attn_weights.register_hook(module.save_attn_gradients)

    def _register_hooks_activations(self):
        for layer in self.target_layers:
            if hasattr(layer, "q_proj"): # is an attention layer
                self.hooks.append(layer.register_forward_hook(self._forward_activate_hooks))
    
    @spaces.GPU(duration=120)
    def generate_cam(self, inputs, tokenizer, temperature, top_p, class_idx=None, visual_pooling_method="CLS", focus="Visual Encoder"):
        """ Generates Grad-CAM heatmap for ViT. """
        
        # Forward pass
        outputs_raw = self.model(**inputs)

        self.model.zero_grad()
        # print(outputs_raw)
        loss = outputs_raw.logits.max(dim=-1).values.sum()

        loss.backward()

        # get image masks
        image_mask = []
        last = 0
        for i in range(inputs["input_ids"].shape[1]):
            decoded_token = tokenizer.decode(inputs["input_ids"][0][i].item())
            if (decoded_token == "<image>"):
                image_mask.append(True)
                last = i
            else:
                image_mask.append(False)


        # Aggregate activations and gradients from ALL layers
        self.activations = [layer.get_attn_map() for layer in self.target_layers]
        self.gradients = [layer.get_attn_gradients() for layer in self.target_layers]
        cam_sum = None
        # Ver 1
        # for act, grad in zip(self.activations, self.gradients):
        #     # act = torch.sigmoid(act)
        #     print("act:", act)
        #     print(len(act))
        #     print("act_shape:", act.shape)
        #     # print("act1_shape:", act[1].shape)
            
        #     act = F.relu(act.mean(dim=1))


        #     # Compute mean of gradients
        #     print("grad:", grad)
        #     print(len(grad))
        #     print("grad_shape:", grad.shape)
        #     grad_weights = grad.mean(dim=1)

        #     print("act shape", act.shape)
        #     print("grad_weights shape", grad_weights.shape)

        #     cam = act * grad_weights
        #     # cam = act
        #     print(cam.shape)

        #     # Sum across all layers
        #     if cam_sum is None:
        #         cam_sum = cam
        #     else:
        #         cam_sum += cam  

        # Ver 2
        for act, grad in zip(self.activations, self.gradients):

            print("act shape", act.shape)
            print("grad shape", grad.shape)

            grad = F.relu(grad)

            cam = act * grad # shape: [1, heads, seq_len, seq_len]
            cam = cam.sum(dim=1) # shape: [1, seq_len, seq_len]

            # Sum across all layers
            if cam_sum is None:
                cam_sum = cam
            else:
                cam_sum += cam 

        cam_sum = F.relu(cam_sum)
        cam_sum = cam_sum.to(torch.float32)

        # thresholding
        # percentile = torch.quantile(cam_sum, 0.4)  # Adjust threshold dynamically
        # cam_sum[cam_sum < percentile] = 0

        # Reshape
        # if visual_pooling_method == "CLS":
        # cam_sum = cam_sum[0, 1:]

        # cam_sum shape: [1, seq_len, seq_len]
        cam_sum_lst = []
        cam_sum_raw = cam_sum
        start_idx = last + 1
        for i in range(start_idx, cam_sum_raw.shape[1]):
            cam_sum = cam_sum_raw[0, i, :] # shape: [1: seq_len]
            # cam_sum_min = cam_sum.min()
            # cam_sum_max = cam_sum.max()
            # cam_sum = (cam_sum - cam_sum_min) / (cam_sum_max - cam_sum_min)
            cam_sum = cam_sum[image_mask].unsqueeze(0) # shape: [1, 1024]
            print("cam_sum shape: ", cam_sum.shape)
            num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
            grid_size = int(num_patches ** 0.5)
            print(f"Detected grid size: {grid_size}x{grid_size}")

            # Fix the reshaping step dynamically
            
            cam_sum = cam_sum.view(grid_size, grid_size)
            cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min())
            cam_sum_lst.append(cam_sum)


        return cam_sum_lst, grid_size
















def generate_gradcam(
    cam, 
    image,
    size = (384, 384),
    alpha=0.5, 
    colormap=cv2.COLORMAP_JET, 
    aggregation='mean',
    normalize=True
):
    """
    Generates a Grad-CAM heatmap overlay on top of the input image.

    Parameters:
      attributions (torch.Tensor): A tensor of shape (C, H, W) representing the
        intermediate activations or gradients at the target layer.
      image (PIL.Image): The original image.
      alpha (float): The blending factor for the heatmap overlay (default 0.5).
      colormap (int): OpenCV colormap to apply (default cv2.COLORMAP_JET).
      aggregation (str): How to aggregate across channels; either 'mean' or 'sum'.

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

