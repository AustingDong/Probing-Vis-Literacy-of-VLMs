import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from demo.cam import generate_gradcam, AttentionGuidedCAMJanus, AttentionGuidedCAMClip
from demo.model_utils import Clip_Utils, Janus_Utils, add_title_to_image

import numpy as np
import matplotlib.pyplot as plt
import gc
from PIL import Image

model_seed = 42
torch.manual_seed(model_seed)
np.random.seed(model_seed)
torch.cuda.manual_seed(model_seed)

model_type = "Janus-1B"
janus_utils = Janus_Utils()
vl_gpt, tokenizer = janus_utils.init_Janus(model_type.split('-')[-1])

clip_utils = Clip_Utils()
clip_utils.init_Clip()

# @torch.inference_mode() # cancel inference, for gradcam
# @spaces.GPU(duration=120) 
# Multimodal Understanding function
def multimodal_understanding(model_type, 
                             saliency_map_method, 
                             visual_pooling_method, 
                             image, question, seed, top_p, temperature, target_token_idx,
                             visualization_layer_min, visualization_layer_max, focus):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    input_text_decoded = ""
    if model_type == "Clip":
        
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
            cam = [generate_gradcam(cam, image, size=(224, 224))]
            grad_cam.remove_hooks()
            target_token_decoded = ""
            answer = ""
    

    elif model_type == "Janus-1B":
        
        for param in vl_gpt.parameters():
            param.requires_grad = True


        prepare_inputs = janus_utils.prepare_inputs(question, image)
        inputs_embeds = janus_utils.generate_inputs_embeddings(prepare_inputs)
        outputs = janus_utils.generate_outputs(inputs_embeds, prepare_inputs, temperature, top_p)

        sequences = outputs.sequences.cpu().tolist()
        answer = tokenizer.decode(sequences[0], skip_special_tokens=True)
        attention_raw = outputs.attentions
        print("answer generated")

        input_ids = prepare_inputs.input_ids[0].cpu().tolist()
        input_ids_decoded = [tokenizer.decode([input_ids[i]]) for i in range(len(input_ids))]
        start=620

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

            gradcam = AttentionGuidedCAMJanus(vl_gpt, target_layers)
            cam_tensors, grid_size = gradcam.generate_cam(prepare_inputs, tokenizer, temperature, top_p, target_token_idx, visual_pooling_method, focus)
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
                        cam_grid = cam_tensor.reshape(24, 24)
                        cam_i = generate_gradcam(cam_grid, image)
                        cam_i = add_title_to_image(cam_i, input_ids_decoded[start + i])

                        cam.append(cam_i)
                        
                    # widths, heights = zip(*(img.size for img in heatmaps))
                    # total_height = sum(heights)
                    # max_width = max(widths)

                    # combined_img = Image.new("RGB", (max_width, total_height))

                    # y_offset = 0
                    # for img in heatmaps:
                    #     combined_img.paste(img, (0, y_offset))  # Stack vertically
                    #     y_offset += img.height
                    # cam = combined_img
                    
                
                    

        elif saliency_map_method == "Attention_Map":
            attn_m_token = attention_raw[target_token_idx]
            img_token_positions = prepare_inputs.images_seq_mask
            mask = img_token_positions[0]

            tg = attn_m_token[1][:, :, :, :len(mask)]
            tg = tg[:, :, :, mask]
            head = 0

            # res = tg[0, head, 0].to(torch.float32)
            res, _ = tg.max(dim=1)
            # res = tg.sum(dim=1)
            res = res.to(torch.float32)
            grid_size = (int)(res.shape[-1] ** 0.5)
            res = res.view(grid_size, grid_size)
            cam = [generate_gradcam(res, image)]
        
        
        # output_arr = output.logits.detach().to(float).to("cpu").numpy()
        # predicted_ids = np.argmax(output_arr, axis=-1) # [1, num_tokens]
        # predicted_ids = predicted_ids.squeeze(0) # [num_tokens]
        # target_token_decoded = tokenizer.decode(predicted_ids[target_token_idx].tolist())


    return answer, cam, input_text_decoded




# Gradio interface

def update_sliders(model):
    if model == "Clip":
        res = (
            gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers min"), 
            gr.Slider(minimum=1, maximum=12, value=12, step=1, label="visualization layers max"),
            gr.Dropdown(choices=["Visual Encoder"], value="Visual Encoder", label="focus")
            )
        return res
    else:
        res = (
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers min"),
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers max"),
            gr.Dropdown(choices=["Visual Encoder", "Language Model"], value="Visual Encoder", label="focus")
        )
        return res

def update_visualization_layers_sliders(focus):
    if focus == "Visual Encoder":
        res = (
            gr.Dropdown(choices=["GradCAM"], value="GradCAM", label="saliency map type"),
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers min"), 
            gr.Slider(minimum=1, maximum=24, value=24, step=1, label="visualization layers max")
        )
        return res
    else:
        res = (
            gr.Dropdown(choices=["GradCAM", "Attention_Map"], value="GradCAM", label="saliency map type"),
            gr.Slider(minimum=1, maximum=24, value=9, step=1, label="visualization layers min"), 
            gr.Slider(minimum=1, maximum=24, value=9, step=1, label="visualization layers max")
        )
        return res

with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image()
            saliency_map_output = gr.Gallery(label="Saliency Map", columns=1)

        with gr.Column():
            model_selector = gr.Dropdown(choices=["Clip", "Janus-1B"], value="Clip", label="model")
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
            fn=update_sliders, 
            inputs=model_selector, 
            outputs=[ 
                visualization_layers_min,
                visualization_layers_max,
                focus
            ]
        )
        
        focus.change(
            fn = update_visualization_layers_sliders,
            inputs = focus,
            outputs=[
                saliency_map_method,
                visualization_layers_min,
                visualization_layers_max,
            ]
        )

        

    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Response")
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
                "images/Treemap.png"
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
                visualization_layers_min, visualization_layers_max, focus],
        outputs=[understanding_output, saliency_map_output, understanding_target_token_decoded_output]
    )
    
demo.launch(share=True)
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")