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
from questions.New_test import new_test_questions
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
model_utils, vl_gpt, tokenizer = None, None, None
model_utils = ChartGemma_Utils()
vl_gpt, tokenizer = model_utils.init_ChartGemma()
for layer in vl_gpt.language_model.model.layers:
    layer.self_attn = ModifiedGemmaAttention(layer.self_attn)
model_name = "ChartGemma-3B"
language_model_max_layer = 24
language_model_best_layer_min = 9
language_model_best_layer_max = 15

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
                             visualization_layer_min, visualization_layer_max, focus, response_type, chart_type, accumulate_method, test_selector):
    # Clear CUDA cache before generating
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # set seed
    set_seed(model_seed=seed)

    input_text_decoded = ""
    answer = ""
    
        
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

    if activation_map_method == "AG-CAM":
        # target_layers = vl_gpt.vision_model.vision_tower.blocks
        
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
        
        # utilize the entire sequence, including <image>s, question, and answer
        entire_inputs = prepare_inputs
        if response_type == "answer + visualization" and focus == "question + answer":
            if model_name.split('-')[0] == "Janus" or model_name.split('-')[0] == "LLaVA":
                entire_inputs = model_utils.prepare_inputs(question, image, answer)
            else:
                entire_inputs["input_ids"] = outputs.sequences
                entire_inputs["attention_mask"] = torch.ones_like(outputs.sequences)
            input_ids = entire_inputs['input_ids'][0].cpu().tolist()
            input_ids_decoded = [tokenizer.decode([input_ids[i]]) for i in range(len(input_ids))]

        cam_tensors, grid_size, start = gradcam.generate_cam(entire_inputs, tokenizer, temperature, top_p, target_token_idx, visual_method, "Language Model", accumulate_method)
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
    FILES_ROOT = f"{RESULTS_ROOT}/{model_name}/{focus}/{visual_method}/{test_selector}/{chart_type}/layer{visualization_layer_min}-{visualization_layer_max}/{'all_tokens' if target_token_idx == -1 else f'--{input_ids_decoded[start + target_token_idx]}--'}"
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
    global model_utils, vl_gpt, tokenizer, clip_utils, model_name, language_model_max_layer, language_model_best_layer_min, language_model_best_layer_max, vision_model_best_layer
    model_name = model_type

    if model_type.split('-')[0] == "Janus":
        # best seed: 70
        clean()
        set_seed()
        model_utils = Janus_Utils()
        vl_gpt, tokenizer = model_utils.init_Janus(model_type.split('-')[-1])
        for layer in vl_gpt.language_model.model.layers:
            layer.self_attn = ModifiedLlamaAttention(layer.self_attn)
        
        language_model_max_layer = 24
        language_model_best_layer_min = 8
        language_model_best_layer_max = 10
        
        sliders = [
            gr.Slider(minimum=1, maximum=24, value=language_model_best_layer_min, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=24, value=language_model_best_layer_max, step=1, label="visualization layers max"),
        ]
        return tuple(sliders)
    
    elif model_type.split('-')[0] == "LLaVA":
        
        clean()
        set_seed()
        model_utils = LLaVA_Utils()
        version = model_type.split('-')[1]
        vl_gpt, tokenizer = model_utils.init_LLaVA(version=version)
        language_model_max_layer = 32 if version == "1.5" else 28
        language_model_best_layer_min = 10
        language_model_best_layer_max = 10

        sliders = [
            gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer_min, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer_max, step=1, label="visualization layers max"),
        ]
        return tuple(sliders)
    
    elif model_type.split('-')[0] == "ChartGemma":
        clean()
        set_seed()
        model_utils = ChartGemma_Utils()
        vl_gpt, tokenizer = model_utils.init_ChartGemma()
        for layer in vl_gpt.language_model.model.layers:
            layer.self_attn = ModifiedGemmaAttention(layer.self_attn)
        language_model_max_layer = 18
        language_model_best_layer_min = 9
        language_model_best_layer_max = 15

        sliders = [
            gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer_min, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer_max, step=1, label="visualization layers max"),
        ]
        return tuple(sliders)



def test_change(test_selector):
    if test_selector == "mini-VLAT":
        return gr.Dataset(
                samples=mini_VLAT_questions,
            )
    elif test_selector == "VLAT":
        return gr.Dataset(
                samples=VLAT_questions,
            )
    elif test_selector == "New_test":
        return gr.Dataset(
                samples=new_test_questions,
            )
    else:
        return gr.Dataset(
                samples=VLAT_old_questions,
            )


with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    
    with gr.Row():
        image_input = gr.Image(height=500, label="Image")
        activation_map_output = gr.Gallery(label="Visualization", height=500, columns=1, preview=True)

    with gr.Row():
        question_input = gr.Textbox(label="Question")
        understanding_output = gr.Textbox(label="Answer")

    with gr.Row():

        with gr.Column():
            model_selector = gr.Dropdown(choices=["ChartGemma-3B", "Janus-Pro-1B", "Janus-Pro-7B", "LLaVA-1.5-7B"], value="ChartGemma-3B", label="model")
            test_selector = gr.Dropdown(choices=["mini-VLAT", "VLAT", "VLAT-old", "New_test"], value="mini-VLAT", label="test")
            chart_type = gr.Textbox(label="Chart Type", value="Any")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
            target_token_idx = gr.Number(label="target_token_idx (-1 means all)", precision=0, value=-1)

        
        with gr.Column():
            response_type = gr.Dropdown(choices=["Visualization only", "answer + visualization"], value="answer + visualization", label="response_type")
            focus = gr.Dropdown(choices=["question", "question + answer"], value="question", label="focus")
            activation_map_method = gr.Dropdown(choices=["AG-CAM"], value="AG-CAM", label="visualization type")
            accumulate_method = gr.Dropdown(choices=["sum", "mult"], value="sum", label="layers accumulate method")
            visual_method = gr.Dropdown(choices=["softmax", "sigmoid"], value="softmax", label="activation function")
            

            visualization_layers_min = gr.Slider(minimum=1, maximum=18, value=11, step=1, label="visualization layers min")
            visualization_layers_max = gr.Slider(minimum=1, maximum=18, value=15, step=1, label="visualization layers max")

        
        


        model_selector.change(
            fn=model_slider_change, 
            inputs=model_selector, 
            outputs=[
                visualization_layers_min,
                visualization_layers_max
            ]
        )

        

    understanding_button = gr.Button("Submit")
    
    understanding_target_token_decoded_output = gr.Textbox(label="Target Token Decoded")


    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        examples=mini_VLAT_questions,
        inputs=[chart_type, question_input, image_input],
    )

    test_selector.change(
        fn=test_change, 
        inputs=test_selector,
        outputs=examples_inpainting.dataset)
    


        
    understanding_button.click(
        multimodal_understanding,
        inputs=[model_selector, activation_map_method, visual_method, image_input, question_input, und_seed_input, top_p, temperature, target_token_idx, 
                visualization_layers_min, visualization_layers_max, focus, response_type, chart_type, accumulate_method, test_selector],
        outputs=[understanding_output, activation_map_output, understanding_target_token_decoded_output]
    )
    
demo.launch(share=True)
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")