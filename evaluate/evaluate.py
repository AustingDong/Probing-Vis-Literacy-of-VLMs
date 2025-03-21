import os
import torch
import numpy as np
from PIL import Image
from demo.model_utils import *
from evaluate.questions import questions

def set_seed(model_seed = 42):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    torch.cuda.manual_seed(model_seed) if torch.cuda.is_available() else None

def clean():
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Frees inter-process CUDA memory
    
    # Empty MacOS Metal backend (if using Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def evaluate(model_type, num_eval = 10):
    for eval_idx in range(num_eval):
        clean()
        set_seed(np.random.randint(0, 1000))
        model_utils, vl_gpt, tokenizer = None, None, None

        if model_type.split('-')[0] == "Janus":
            model_utils = Janus_Utils()
            vl_gpt, tokenizer = model_utils.init_Janus(model_type.split('-')[-1])

        elif model_type.split('-')[0] == "LLaVA":
            model_utils = LLaVA_Utils()
            version = model_type.split('-')[1]
            vl_gpt, tokenizer = model_utils.init_LLaVA(version=version)
        
        elif model_type.split('-')[0] == "ChartGemma":
            model_utils = ChartGemma_Utils()
            vl_gpt, tokenizer = model_utils.init_ChartGemma()

        for question in questions:
            chart_type = question[0]
            q = question[1]
            img_path = question[2]
            image = np.array(Image.open(img_path).convert("RGB"))

            prepare_inputs = model_utils.prepare_inputs(q, image)
            temperature = 0.9
            top_p = 0.1

            if model_type.split('-')[0] == "Janus":
                inputs_embeds = model_utils.generate_inputs_embeddings(prepare_inputs)
                outputs = model_utils.generate_outputs(inputs_embeds, prepare_inputs, temperature, top_p)
            else:
                outputs = model_utils.generate_outputs(prepare_inputs, temperature, top_p)

            sequences = outputs.sequences.cpu().tolist()
            answer = tokenizer.decode(sequences[0], skip_special_tokens=True)

            RESULTS_ROOT = "./evaluate/results"
            FILES_ROOT = f"{RESULTS_ROOT}/{model_type}/{eval_idx}"
            os.makedirs(FILES_ROOT, exist_ok=True)

            with open(f"{FILES_ROOT}/{chart_type}.txt", "w") as f:
                f.write(answer)
                f.close()



if __name__ == '__main__':
    
    # models = ["ChartGemma", "Janus-Pro-1B", "Janus-Pro-7B", "LLaVA-1.5-7B"]
    # models = ["ChartGemma", "Janus-Pro-1B"]
    # models = ["Janus-Pro-7B", "LLaVA-1.5-7B"]
    models = ["LLaVA-1.5-7B"]
    for model_type in models:
        evaluate(model_type=model_type, num_eval=10)
