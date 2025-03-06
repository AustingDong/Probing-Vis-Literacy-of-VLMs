import torch
import numpy as np
import spaces
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoModelForCausalLM, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, LlavaNextProcessor, AutoProcessor, PaliGemmaForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from janus.models import MultiModalityCausalLM, VLChatProcessor

@spaces.GPU(duration=120)
def set_dtype_device(model, precision=16):
    dtype = (torch.bfloat16 if torch.cuda.is_available() else torch.float16) if precision==16 else (torch.bfloat32 if torch.cuda.is_available() else torch.float32)
    cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model = model.to(dtype).cuda()
    else:
        torch.set_default_device("cpu")
        model = model.to(dtype)
    return model, dtype, cuda_device


class Model_Utils:
    def __init__(self):
        pass
    
    @spaces.GPU(duration=120)
    def prepare_inputs(self):
        raise NotImplementedError
    
    @spaces.GPU(duration=120)
    def generate_outputs(self):
        raise NotImplementedError



class Clip_Utils(Model_Utils):
    def __init__(self):
        self.edge = 224
        super().__init__()

    def init_Clip(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.processor.feature_extractor.size = {"height": self.edge, "width": self.edge}

    @spaces.GPU(duration=120)
    def prepare_inputs(self, question_lst, image):
        image = Image.fromarray(image)
        # print("image_size: ", image.size)
        inputs = self.processor(text=question_lst, images=image, return_tensors="pt", padding=True)
        return inputs
        

class Janus_Utils(Model_Utils):
    def __init__(self):
        super().__init__()

    def init_Janus(self, num_params="1B"):

        model_path = f"deepseek-ai/Janus-Pro-{num_params}"
        config = AutoConfig.from_pretrained(model_path)
        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                                    language_config=language_config,
                                                    trust_remote_code=True,
                                                    ignore_mismatched_sizes=True,
                                                    )
        self.vl_gpt, self.dtype, self.cuda_device = set_dtype_device(self.vl_gpt)
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        return self.vl_gpt, self.tokenizer
    
    @spaces.GPU(duration=120)
    def prepare_inputs(self, question, image):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        pil_images = [Image.fromarray(image)]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.cuda_device, dtype=self.dtype)

        return prepare_inputs
    
    @spaces.GPU(duration=120)
    def generate_inputs_embeddings(self, prepare_inputs):
        return self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    @spaces.GPU(duration=120)
    def generate_outputs(self, inputs_embeds, prepare_inputs, temperature, top_p, with_attn=False):
        
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False if temperature == 0 else True,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_attentions=True
        )

        return outputs

class LLaVA_Utils(Model_Utils):
    def __init__(self):
        super().__init__()

    def init_LLaVA(self):

        # model_path = "llava-hf/llava-1.5-7b-hf"
        model_path = "llava-hf/llava-v1.6-vicuna-7b-hf"
        config = AutoConfig.from_pretrained(model_path)

        self.vl_gpt = LlavaNextForConditionalGeneration.from_pretrained(model_path,
                                                    low_cpu_mem_usage=True,
                                                    attn_implementation = 'eager',
                                                    output_attentions=True
                                                    )
        self.vl_gpt, self.dtype, self.cuda_device = set_dtype_device(self.vl_gpt)
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        return self.vl_gpt, self.tokenizer
    
    @spaces.GPU(duration=120)
    def prepare_inputs(self, question, image):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        pil_images = [Image.fromarray(image)]
        prepare_inputs = self.processor(
            images=pil_images, text=prompt, return_tensors="pt"
        ).to(self.cuda_device, dtype=self.dtype)

        return prepare_inputs
    
    @spaces.GPU(duration=120)
    def generate_inputs_embeddings(self, prepare_inputs):
        return self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    @spaces.GPU(duration=120)
    def generate_outputs(self, prepare_inputs, temperature, top_p):
        
        outputs = self.vl_gpt.generate(
            **prepare_inputs,
            max_new_tokens=512,
            do_sample=False if temperature == 0 else True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        return outputs
    




class ChartGemma_Utils(Model_Utils):
    def __init__(self):
        super().__init__()

    def init_ChartGemma(self):

        model_path = "ahmed-masry/chartgemma"
        

        self.vl_gpt = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,  
            attn_implementation="eager", 
            output_attentions=True
        )
        self.vl_gpt, self.dtype, self.cuda_device = set_dtype_device(self.vl_gpt)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        return self.vl_gpt, self.tokenizer
    
    @spaces.GPU(duration=120)
    def prepare_inputs(self, question, image):

        pil_image = Image.fromarray(image)
        prepare_inputs = self.processor(
            images=pil_image, text=[question], return_tensors="pt"
        ).to(self.cuda_device, dtype=self.dtype)

        return prepare_inputs
    
    @spaces.GPU(duration=120)
    def generate_inputs_embeddings(self, prepare_inputs):
        return self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    @spaces.GPU(duration=120)
    def generate_outputs(self, prepare_inputs, temperature, top_p):
        
        outputs = self.vl_gpt.generate(
            **prepare_inputs,
            max_new_tokens=512,
            do_sample=False if temperature == 0 else True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        return outputs




def add_title_to_image(image, title, font_size=50):
    """Adds a title above an image using PIL and textbbox()."""
    img_width, img_height = image.size

    # Create a blank image for title
    title_height = font_size + 10  # Some padding
    title_image = Image.new("RGB", (img_width, title_height), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(title_image)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Use Arial if available
    except:
        font = ImageFont.load_default(font_size)  # Use default if Arial not found

    # Get text size (updated for PIL >= 10)
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center the title
    text_position = ((img_width - text_width) // 2, (title_height - text_height) // 2)

    draw.text(text_position, title, fill="black", font=font)

    # Concatenate title with image
    combined = Image.new("RGB", (img_width, img_height + title_height))
    combined.paste(title_image, (0, 0))  # Place title at the top
    combined.paste(image, (0, title_height))  # Place original image below

    return combined
    

    