import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from demo.cam import generate_gradcam, AttentionGuidedCAMJanus, AttentionGuidedCAMClip, AttentionGuidedCAMChartGemma, AttentionGuidedCAMLLaVA
from demo.model_utils import Clip_Utils, Janus_Utils, LLaVA_Utils, ChartGemma_Utils, add_title_to_image

import numpy as np
import matplotlib.pyplot as plt
import gc
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
                             saliency_map_method, 
                             visual_pooling_method, 
                             image, question, seed, top_p, temperature, target_token_idx,
                             visualization_layer_min, visualization_layer_max, focus, response_type):
    # Clear CUDA cache before generating
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None

    input_text_decoded = ""
    answer = ""
    if model_name == "Clip":
        
        inputs = clip_utils.prepare_inputs([question], image)


        if saliency_map_method == "GradCAM":
            # Generate Grad-CAM
            all_layers = [layer.layer_norm1 for layer in clip_utils.model.vision_model.encoder.layers]
            if visualization_layers_min.value != visualization_layers_max.value:
                target_layers = all_layers[visualization_layer_min-1 : visualization_layer_max-1]
            else:
                target_layers = [all_layers[visualization_layer_min-1]]
            grad_cam = AttentionGuidedCAMClip(clip_utils.model, target_layers)
            cam, outputs, grid_size = grad_cam.generate_cam(inputs, class_idx=0, visual_pooling_method=visual_pooling_method)
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
        if model_name.split('-')[0] == "Janus":
            start = 620 
        elif model_name.split('-')[0] == "ChartGemma":
            start = 1024
        else: 
            start = 512

        if saliency_map_method == "GradCAM":
            # target_layers = vl_gpt.vision_model.vision_tower.blocks
            if focus == "Visual Encoder":
                all_layers = [block.norm1 for block in vl_gpt.vision_model.vision_tower.blocks]
            else:
                all_layers = [layer.self_attn for layer in vl_gpt.language_model.model.layers]

            if visualization_layers_min.value != visualization_layers_max.value:
                target_layers = all_layers[visualization_layer_min-1 : visualization_layer_max-1]
            else:
                target_layers = [all_layers[visualization_layer_min-1]]

            if model_name.split('-')[0] == "Janus":
                gradcam = AttentionGuidedCAMJanus(vl_gpt, target_layers)
            elif model_name.split('-')[0] == "LLaVA":
                gradcam = AttentionGuidedCAMLLaVA(vl_gpt, target_layers)
            elif model_name.split('-')[0] == "ChartGemma":
                gradcam = AttentionGuidedCAMChartGemma(vl_gpt, target_layers)

            cam_tensors, grid_size = gradcam.generate_cam(prepare_inputs, tokenizer, temperature, top_p, target_token_idx, visual_pooling_method, focus)
            gradcam.remove_hooks()


            if focus == "Visual Encoder":
                cam_grid = cam_tensors.reshape(grid_size, grid_size)
                cam = [generate_gradcam(cam_grid, image)]
            else:
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

    return answer, cam, input_text_decoded




# Gradio interface

def model_slider_change(model_type):
    global model_utils, vl_gpt, tokenizer, clip_utils, model_name, language_model_max_layer, language_model_best_layer
    model_name = model_type
    if model_type == "Clip":
        clean()
        set_seed()
        clip_utils = Clip_Utils()
        clip_utils.init_Clip()
        res = (
            gr.Dropdown(choices=["Visualization only"], value="Visualization only", label="response_type"),
            gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers max"),
            gr.Dropdown(choices=["Visual Encoder"], value="Visual Encoder", label="focus"),
            gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type")
        )
        return res
    elif model_type.split('-')[0] == "Janus":
        
        clean()
        set_seed()
        model_utils = Janus_Utils()
        vl_gpt, tokenizer = model_utils.init_Janus(model_type.split('-')[-1])
        language_model_max_layer = 24
        language_model_best_layer = 8

        res = (
            gr.Dropdown(choices=["Visualization only", "answer + visualization"], value="Visualization only", label="response_type"),
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers max"),
            gr.Dropdown(choices=["Visual Encoder", "Language Model"], value="Visual Encoder", label="focus"),
            gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type")
        )
        return res
    
    elif model_type.split('-')[0] == "LLaVA":
        
        clean()
        set_seed()
        model_utils = LLaVA_Utils()
        vl_gpt, tokenizer = model_utils.init_LLaVA()
        language_model_max_layer = 32
        language_model_best_layer = 24

        res = (
            gr.Dropdown(choices=["Visualization only", "answer + visualization"], value="Visualization only", label="response_type"),
            gr.Slider(minimum=1, maximum=32, value=24, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=32, value=24, step=1, label="visualization layers max"),
            gr.Dropdown(choices=["Language Model"], value="Language Model", label="focus"),
            gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type")
        )
        return res
    
    elif model_type.split('-')[0] == "ChartGemma":
        clean()
        set_seed()
        model_utils = ChartGemma_Utils()
        vl_gpt, tokenizer = model_utils.init_ChartGemma()
        language_model_max_layer = 18
        language_model_best_layer = 15

        res = (
            gr.Dropdown(choices=["Visualization only", "answer + visualization"], value="Visualization only", label="response_type"),
            gr.Slider(minimum=1, maximum=18, value=15, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=18, value=15, step=1, label="visualization layers max"),
            gr.Dropdown(choices=["Language Model"], value="Language Model", label="focus"),
            gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type")
        )
        return res

    


def focus_change(focus):
    global model_name, language_model_max_layer
    if model_name == "Clip":
        res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type"),
                gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers max")
            )
        return res

    if focus == "Language Model":
        if response_type.value == "answer + visualization":
            res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type"),
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers max")
            )
            return res
        else:
            res = (
                gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type"),
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers min"), 
                gr.Slider(minimum=1, maximum=language_model_max_layer, value=language_model_best_layer, step=1, label="visualization layers max")
            )
            return res

    else:
        res = (
            gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type"),
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers min"), 
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers max")
        )
        return res





with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image()
            saliency_map_output = gr.Gallery(label="Saliency Map", height=300, columns=1)

        with gr.Column():
            model_selector = gr.Dropdown(choices=["Clip", "ChartGemma-2B", "Janus-1B", "Janus-7B", "LLaVA-v1.6-Mistral-7B"], value="Clip", label="model")
            response_type = gr.Dropdown(choices=["Visualization only"], value="Visualization only", label="response_type")
            focus = gr.Dropdown(choices=["Visual Encoder"], value="Visual Encoder", label="focus")
            saliency_map_method = gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type")
            visual_pooling_method = gr.Dropdown(choices=["CLS", "max", "avg"], value="CLS", label="visual pooling method")
            

            visualization_layers_min = gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers min")
            visualization_layers_max = gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers max")
        
            question_input = gr.Textbox(label="Question")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
            target_token_idx = gr.Number(label="target_token_idx (-1 means all)", precision=0, value=-1)
        


        model_selector.change(
            fn=model_slider_change, 
            inputs=model_selector, 
            outputs=[
                response_type,
                visualization_layers_min,
                visualization_layers_max,
                focus,
                saliency_map_method
            ]
        )
        
        focus.change(
            fn = focus_change,
            inputs = focus,
            outputs=[
                saliency_map_method,
                visualization_layers_min,
                visualization_layers_max,
            ]
        )

        # response_type.change(
        #     fn = response_type_change,
        #     inputs = response_type,
        #     outputs = [saliency_map_method]
        # )

        

    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Answer")
    understanding_target_token_decoded_output = gr.Textbox(label="Target Token Decoded")


    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        examples=[
            
            [
                "What is the approximate global smartphone market share of Samsung?",
                "images/PieChart.png"
            ],
            [
                "What is the average internet speed in Japan?",
                "images/BarChart.png"
            ],
            [
                "What was the average price of coffee beans in October 2019?",
                "images/AreaChart.png"
            ],
            [
                "Which city's metro system has the largest number of stations?", 
                "images/BubbleChart.png"
            ],

            [ 
                "True/False: In 2020, the unemployment rate for Washington (WA) was higher than that of Wisconsin (WI).", 
                "images/Choropleth_New.png"
            ],

            [ 
                "What distance have customers traveled in the taxi the most?", 
                "images/Histogram.png"
            ],

            [
                "What was the price of a barrel of oil in February 2020?", 
                "images/LineChart.png" 
            ],

            [
                "True/False: eBay is nested in the Software category.", 
                "images/TreeMap.png"
            ],

            [
                "True/False: There is a negative linear relationship between the height and the weight of the 85 males.", 
                "images/Scatterplot.png"
            ],
            
            [ 
                "Which country has the lowest proportion of Gold medals?", 
                "images/Stacked100.png"
            ],

            [
                "What was the ratio of girls named 'Isla' to girls named 'Amelia' in 2012 in the UK?", 
                "images/StackedArea.png"
            ],

            [
                "What is the cost of peanuts in Seoul?", 
                "images/StackedBar.png"
            ],

            [
                "Where is the dog? Left or Right?",
                "images/cat_dog.png"
            ]
            

            # [
            #     "explain this meme",
            #     "images/doge.png",
            # ],
            # [
            #     "Convert the formula into latex code.",
            #     "images/equation.png",
            # ],
            
        ],
        inputs=[question_input, image_input],
    )
    


        
    understanding_button.click(
        multimodal_understanding,
        inputs=[model_selector, saliency_map_method, visual_pooling_method, image_input, question_input, und_seed_input, top_p, temperature, target_token_idx, 
                visualization_layers_min, visualization_layers_max, focus, response_type],
        outputs=[understanding_output, saliency_map_output, understanding_target_token_decoded_output]
    )
    
demo.launch(share=True)
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")