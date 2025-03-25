import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from demo.visualization import generate_gradcam, VisualizationJanus, VisualizationClip, VisualizationChartGemma, VisualizationLLaVA
from demo.model_utils import Clip_Utils, Janus_Utils, LLaVA_Utils, ChartGemma_Utils, add_title_to_image
from demo.modified_attn import ModifiedLlamaAttention, ModifiedGemmaAttention
from questions.mini_VLAT import mini_VLAT_questions
from questions.VLAT_old import VLAT_old_questions
from questions.VLAT import VLAT_questions
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import spaces
from PIL import Image

def set_seed(model_seed = 42):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    torch.cuda.manual_seed(model_seed) if torch.cuda.is_available() else None

set_seed()
clip_utils = Clip_Utils()
clip_utils.init_Clip()
model_utils, vl_gpt, tokenizer = None, None, None
model_name = "Clip"
language_model_max_layer = 24
language_model_best_layer = 8
vision_model_best_layer = 24

def clean():
    global model_utils, vl_gpt, tokenizer, clip_utils
    # Move models to CPU first (prevents CUDA references)
    if 'vl_gpt' in globals() and vl_gpt is not None:
        vl_gpt.to("cpu")
    if 'clip_utils' in globals() and clip_utils is not None:
        del clip_utils

    # Delete all references
    del model_utils, vl_gpt, tokenizer
    model_utils, vl_gpt, tokenizer, clip_utils = None, None, None, None
    gc.collect()

    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Frees inter-process CUDA memory
    
    # Empty MacOS Metal backend (if using Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# Multimodal Understanding function
@spaces.GPU(duration=120)
def multimodal_understanding(model_type, 
                             activation_map_method, 
                             visual_method, 
                             image, question, seed, top_p, temperature, target_token_idx,
                             visualization_layer_min, visualization_layer_max, focus, response_type, chart_type, accumulate_method):
    # Clear CUDA cache before generating
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # set seed
    set_seed(model_seed=seed)

    input_text_decoded = ""
    answer = ""
    if model_name == "Clip":
        
        inputs = clip_utils.prepare_inputs([question], image)


        if activation_map_method == "GradCAM":
            # Generate Grad-CAM
            all_layers = [layer.layer_norm1 for layer in clip_utils.model.vision_model.encoder.layers]

            if visualization_layer_min != visualization_layer_max:
                target_layers = all_layers[visualization_layer_min-1 : visualization_layer_max-1]
            else:
                target_layers = [all_layers[visualization_layer_min-1]]
            grad_cam = VisualizationClip(clip_utils.model, target_layers)
            cam, outputs, grid_size = grad_cam.generate_cam(inputs, target_token_idx=0, visual_method=visual_method)
            cam = cam.to("cpu")
            cam = [generate_gradcam(cam, image, size=(224, 224))]
            grad_cam.remove_hooks()
            target_token_decoded = ""
            
    

    else:
        
        for param in vl_gpt.parameters():
            param.requires_grad = True


        prepare_inputs = model_utils.prepare_inputs(question, image)

        if response_type == "answer + visualization":
            if model_name.split('-')[0] == "Janus":
                inputs_embeds = model_utils.generate_inputs_embeddings(prepare_inputs)
                outputs = model_utils.generate_outputs(inputs_embeds, prepare_inputs, temperature, top_p)
            else:
                outputs = model_utils.generate_outputs(prepare_inputs, temperature, top_p)

            sequences = outputs.sequences.cpu().tolist()
            answer = tokenizer.decode(sequences[0], skip_special_tokens=True)
            attention_raw = outputs.attentions
            print("answer generated")

        input_ids = prepare_inputs.input_ids[0].cpu().tolist()
        input_ids_decoded = [tokenizer.decode([input_ids[i]]) for i in range(len(input_ids))]

        if activation_map_method == "GradCAM":
            # target_layers = vl_gpt.vision_model.vision_tower.blocks
            if focus == "Visual Encoder":
                if model_name.split('-')[0] == "Janus":
                    all_layers = [block.norm1 for block in vl_gpt.vision_model.vision_tower.blocks]
                else:
                    all_layers = [block.layer_norm1 for block in vl_gpt.vision_tower.vision_model.encoder.layers]
            else:
                all_layers = [layer.self_attn for layer in vl_gpt.language_model.model.layers]
            
            print("layer values:", visualization_layer_min, visualization_layer_max)
            if visualization_layer_min != visualization_layer_max:
                print("multi layers")
                target_layers = all_layers[visualization_layer_min-1 : visualization_layer_max]
            else:
                print("single layer")
                target_layers = [all_layers[visualization_layer_min-1]]
            

            if model_name.split('-')[0] == "Janus":
                gradcam = VisualizationJanus(vl_gpt, target_layers)
            elif model_name.split('-')[0] == "LLaVA":
                gradcam = VisualizationLLaVA(vl_gpt, target_layers)
            elif model_name.split('-')[0] == "ChartGemma":
                gradcam = VisualizationChartGemma(vl_gpt, target_layers)

            start = 0
            cam = []
            if focus == "Visual Encoder":
                if target_token_idx != -1:
                    cam_tensors, grid_size, start = gradcam.generate_cam(prepare_inputs, tokenizer, temperature, top_p, target_token_idx, visual_method, focus)
                    cam_grid = cam_tensors.reshape(grid_size, grid_size)
                    cam_i = generate_gradcam(cam_grid, image)
                    cam_i = add_title_to_image(cam_i, input_ids_decoded[start + target_token_idx])
                    cam = [cam_i]
                else:
                    i = 0
                    cam = []
                    while start + i < len(input_ids_decoded):
                        if model_name.split('-')[0] == "Janus":
                            gradcam = VisualizationJanus(vl_gpt, target_layers)
                        elif model_name.split('-')[0] == "LLaVA":
                            gradcam = VisualizationLLaVA(vl_gpt, target_layers)
                        elif model_name.split('-')[0] == "ChartGemma":
                            gradcam = VisualizationChartGemma(vl_gpt, target_layers)
                        cam_tensors, grid_size, start = gradcam.generate_cam(prepare_inputs, tokenizer, temperature, top_p, i, visual_method, focus, accumulate_method)
                        cam_grid = cam_tensors.reshape(grid_size, grid_size)
                        cam_i = generate_gradcam(cam_grid, image)
                        cam_i = add_title_to_image(cam_i, input_ids_decoded[start + i])
                        cam.append(cam_i)
                        gradcam.remove_hooks()
                        i += 1
            else:
                cam_tensors, grid_size, start = gradcam.generate_cam(prepare_inputs, tokenizer, temperature, top_p, target_token_idx, visual_method, focus, accumulate_method)
                if target_token_idx != -1:
                    input_text_decoded = input_ids_decoded[start + target_token_idx]
                    for i, cam_tensor in enumerate(cam_tensors):
                        if i == target_token_idx:
                            cam_grid = cam_tensor.reshape(grid_size, grid_size)
                            cam_i = generate_gradcam(cam_grid, image)
                            cam = [add_title_to_image(cam_i, input_text_decoded)]
                            break
                else:
                    cam = []
                    for i, cam_tensor in enumerate(cam_tensors):
                        cam_grid = cam_tensor.reshape(grid_size, grid_size)
                        cam_i = generate_gradcam(cam_grid, image)
                        cam_i = add_title_to_image(cam_i, input_ids_decoded[start + i])

                        cam.append(cam_i)
                     
            gradcam.remove_hooks()
                

    # Collect Results
    RESULTS_ROOT = "./results"
    FILES_ROOT = f"{RESULTS_ROOT}/{model_name}/{focus}/{visual_method}/{chart_type}/layer{visualization_layer_min}-{visualization_layer_max}/{'all_tokens' if target_token_idx == -1 else f'--{input_ids_decoded[start + target_token_idx]}--'}"
    os.makedirs(FILES_ROOT, exist_ok=True)
    
    for i, cam_p in enumerate(cam):
        cam_p.save(f"{FILES_ROOT}/{i}.png")
            
    with open(f"{FILES_ROOT}/input_text_decoded.txt", "w") as f:
        f.write(input_text_decoded)
        f.close()

    with open(f"{FILES_ROOT}/answer.txt", "w") as f:
        f.write(answer)
        f.close()
            


    return answer, cam, input_text_decoded




# Gradio interface

def model_slider_change(model_type):
    global model_utils, vl_gpt, tokenizer, clip_utils, model_name, language_model_max_layer, language_model_best_layer, vision_model_best_layer
    model_name = model_type


    encoder_only_res = [
        gr.Dropdown(choices=["Visualization only"], value="Visualization only", label="response_type"),
        gr.Dropdown(choices=["Visual Encoder"], value="Visual Encoder", label="focus"),
        gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
        gr.Dropdown(choices=["CLS", "max", "avg"], value="CLS", label="visual pooling method")
    ]

    visual_res = [
        gr.Dropdown(choices=["Visualization only", "answer + visualization"], value="Visualization only", label="response_type"),
        gr.Dropdown(choices=["Visual Encoder"], value="Visual Encoder", label="focus"),
        gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
        gr.Dropdown(choices=["softmax", "sigmoid"], value="softmax", label="activation function")
    ]

    language_res = [
        gr.Dropdown(choices=["Visualization only", "answer + visualization"], value="answer + visualization", label="response_type"),
        gr.Dropdown(choices=["Language Model"], value="Language Model", label="focus"),
        gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
        gr.Dropdown(choices=["softmax", "sigmoid"], value="softmax", label="activation function")
    ]


    if model_type == "Clip":
        clean()
        set_seed()
        clip_utils = Clip_Utils()
        clip_utils.init_Clip()
        sliders = [
            gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers max"),
        ]
        return tuple(encoder_only_res + sliders)
    
    elif model_type.split('-')[0] == "Janus":
        
        clean()
        set_seed()
        model_utils = Janus_Utils()
        vl_gpt, tokenizer = model_utils.init_Janus(model_type.split('-')[-1])
        for layer in vl_gpt.language_model.model.layers:
            layer.self_attn = ModifiedLlamaAttention(layer.self_attn)
        
        language_model_max_layer = 24
        language_model_best_layer = 8
        
        sliders = [
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers max"),
        ]
        return tuple(visual_res + sliders)
    
    elif model_type.split('-')[0] == "LLaVA":
        
        clean()
        set_seed()
        model_utils = LLaVA_Utils()
        version = model_type.split('-')[1]
        vl_gpt, tokenizer = model_utils.init_LLaVA(version=version)
        language_model_max_layer = 32 if version == "1.5" else 28
        language_model_best_layer = 10

        sliders = [
            gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers max"),
        ]
        return tuple(language_res + sliders)
    
    elif model_type.split('-')[0] == "ChartGemma":
        clean()
        set_seed()
        model_utils = ChartGemma_Utils()
        vl_gpt, tokenizer = model_utils.init_ChartGemma()
        for layer in vl_gpt.language_model.model.layers:
            layer.self_attn = ModifiedGemmaAttention(layer.self_attn)
        language_model_max_layer = 18
        vision_model_best_layer = 19
        language_model_best_layer = 15

        sliders = [
            gr.Slider(minimum=1, maximum=language_model_best_layer, value=language_model_best_layer, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=language_model_best_layer, value=language_model_best_layer, step=1, label="visualization layers max"),
        ]
        return tuple(language_res + sliders)

    


def focus_change(focus):
    global model_name, language_model_max_layer
    if model_name == "Clip":
        res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
                gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers max")
            )
        return res

    if focus == "Language Model":
        if response_type.value == "answer + visualization":
            res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers max")
            )
            return res
        else:
            res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers max")
            )
            return res

    else:
        if model_name.split('-')[0] == "ChartGemma":
            res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
                gr.Slider(minimum=1, maximum=26, value=vision_model_best_layer, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=26, value=vision_model_best_layer, step=1, label="visualization layers max")
            )
            return res
        
        else:
            res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="activation map type"),
                gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers max")
            )
            return res





with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    
    with gr.Row():
        image_input = gr.Image(height=500, label="Image")
        activation_map_output = gr.Gallery(label="Visualization", height=500, columns=1, preview=True)

    with gr.Row():
        chart_type = gr.Textbox(label="Chart Type")
        understanding_output = gr.Textbox(label="Answer")

    with gr.Row():

        with gr.Column():
            model_selector = gr.Dropdown(choices=["Clip", "ChartGemma-3B", "Janus-Pro-1B", "Janus-Pro-7B", "LLaVA-1.5-7B"], value="Clip", label="model")
            question_input = gr.Textbox(label="Input Prompt")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
            target_token_idx = gr.Number(label="target_token_idx (-1 means all)", precision=0, value=-1)

        
        with gr.Column():
            response_type = gr.Dropdown(choices=["Visualization only"], value="Visualization only", label="response_type")
            focus = gr.Dropdown(choices=["Visual Encoder"], value="Visual Encoder", label="focus")
            activation_map_method = gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="visualization type")
            accumulate_method = gr.Dropdown(choices=["sum", "mult"], value="sum", label="layers accumulate method")
            visual_method = gr.Dropdown(choices=["CLS", "max", "avg"], value="CLS", label="visual pooling method")
            

            visualization_layers_min = gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers min")
            visualization_layers_max = gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers max")

        
        


        model_selector.change(
            fn=model_slider_change, 
            inputs=model_selector, 
            outputs=[
                response_type,
                focus,
                activation_map_method,
                visual_method,
                visualization_layers_min,
                visualization_layers_max
            ]
        )
        
        focus.change(
            fn = focus_change,
            inputs = focus,
            outputs=[
                activation_map_method,
                visualization_layers_min,
                visualization_layers_max,
            ]
        )

        

    understanding_button = gr.Button("Submit")
    
    understanding_target_token_decoded_output = gr.Textbox(label="Target Token Decoded")


    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        # examples=mini_VLAT_questions,
        examples=VLAT_questions,
        inputs=[chart_type, question_input, image_input],
    )
    


        
    understanding_button.click(
        multimodal_understanding,
        inputs=[model_selector, activation_map_method, visual_method, image_input, question_input, und_seed_input, top_p, temperature, target_token_idx, 
                visualization_layers_min, visualization_layers_max, focus, response_type, chart_type, accumulate_method],
        outputs=[understanding_output, activation_map_output, understanding_target_token_decoded_output]
    )
    
demo.launch(share=True)
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")