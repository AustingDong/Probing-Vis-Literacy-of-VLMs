import os
import torch
import base64
import json
import numpy as np
from PIL import Image
from openai import OpenAI
from demo.model_utils import *
from pydantic import BaseModel


questions = json.load(open("evaluate/new_test.json", "r"))
judge_client = OpenAI(api_key=os.environ["GEMINI_HCI_API_KEY"], 
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

class Judge_Result(BaseModel):
    result: int

def set_seed(model_seed = 70):
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed) if torch.cuda.is_available() else None

def clean():
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Frees inter-process CUDA memory
    
    # Empty MacOS Metal backend (if using Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def llm_judge(answer, options, correct_answer):
    completion = judge_client.beta.chat.completions.parse(
        model="gemini-2.5-pro-preview-03-25",
        messages=[
            { "role": "system", "content": "You are a judge that evaluates the correctness of answers to questions. The answer might not be the letter of correct option. You need to judge correctness of the answer, comparing to the options and correct option. Return the correctness in 1: Correct or 0: Incorrect." },
            { "role": "user", "content": f":Options: {options}\nAnswer:{answer},Correct Option: {correct_answer}" },
        ],
        response_format=Judge_Result
    )
    answer = completion.choices[0].message.content
    print(f"Judge Answer: {answer}")
    return json.loads(answer)["result"]
    

def evaluate(model_type, num_eval = 10):
    sum_correct = np.zeros(len(questions))
    RESULTS_ROOT = "./evaluate/results"
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
        
        elif model_type.split('-')[0] == "GPT":
            client = OpenAI(api_key=os.environ["OPENAI_HCI_API_KEY"])
        
        elif model_type.split('-')[0] == "Gemini":
            client = OpenAI(api_key=os.environ["GEMINI_HCI_API_KEY"], 
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

        for question_idx, question in enumerate(questions):
            chart_type = question["type"]
            q = question["question"]
            img_path = question["img_path"]
            options = question.get("options", None)
            correct_answer = question.get("correct_answer", None)

            image = np.array(Image.open(img_path).convert("RGB"))

            
            input_text = f"Options: {options}\nQuestion: {q}\n"
            if model_type.split('-')[0] == "GPT":
                base64_image = encode_image(img_path)
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                { "type": "text", "text": f"{input_text}" },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                )
                answer = completion.choices[0].message.content

            elif model_type.split('-')[0] == "Gemini":
                base64_image = encode_image(img_path)
                completion = client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                { "type": "text", "text": f"{input_text}" },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                )
                answer = completion.choices[0].message.content

            else:
                
                prepare_inputs = model_utils.prepare_inputs(input_text, image)
                temperature = 0.1
                top_p = 0.95

                if model_type.split('-')[0] == "Janus":
                    inputs_embeds = model_utils.generate_inputs_embeddings(prepare_inputs)
                    outputs = model_utils.generate_outputs(inputs_embeds, prepare_inputs, temperature, top_p)
                else:
                    outputs = model_utils.generate_outputs(prepare_inputs, temperature, top_p)

                sequences = outputs.sequences.cpu().tolist()
                answer = tokenizer.decode(sequences[0], skip_special_tokens=True)

            # Judge the answer
            result_judge = llm_judge(answer, options, correct_answer)
            sum_correct[question_idx] += 1 if result_judge else 0
            print(f"Model: {model_type}, Question: {question_idx + 1}, Answer: {answer}, Correct: {result_judge}")

            # Save the results
            FILES_ROOT = f"{RESULTS_ROOT}/{model_type}/{eval_idx}"
            os.makedirs(FILES_ROOT, exist_ok=True)

            with open(f"{FILES_ROOT}/Q{question_idx + 1}-{chart_type}.txt", "w") as f:
                f.write(answer)
                f.close()
    accuracy = sum_correct / num_eval
    print(f"Model: {model_type}, Accuracy: {accuracy}")
    with open(f"{RESULTS_ROOT}/{model_type}/accuracy.txt", "w") as f:
        for question_idx, question in enumerate(questions):
            chart_type = question["type"]
            f.write(f"Chart Type: {chart_type}, Accuracy: {accuracy[question_idx]}\n")
        f.close()

if __name__ == '__main__':

    # models = ["Janus-Pro-1B", "ChartGemma", "GPT-4o", "Gemini-2.0-flash", "Janus-Pro-7B", "LLaVA-1.5-7B"]
    models = ["LLaVA-1.5-7B", "Gemini-2.0-flash", "Janus-Pro-7B", ]

    for model_type in models:
        evaluate(model_type=model_type, num_eval=10)
