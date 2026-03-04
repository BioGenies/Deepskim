# Binary classifier threshold tuner (uses validation set to find optimal threshold)
from argparse import ArgumentParser
import yaml 
from collections import namedtuple
import numpy as np
from copy import deepcopy

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from torch.amp import autocast
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


from data.dataset_qlora import prepare_dataset
from training.qlora import compute_metrics
from helpers.biomistral import BioMistralInference
from training.qlora import compute_metrics
from utils.evaluation import convert_scores_to_probs, gather_yes_no_logprobs, find_best_threshold

def tokenize_fn_val(example, tokenizer):
    prompt = example["prompt"]

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if prompt_ids[-1] == tokenizer.eos_token:
        prompt_ids = prompt_ids[:-1]

    input_ids = prompt_ids 


    # Create labels (mask out the prompt)
    labels = [-100] * len(prompt_ids) 

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def evaluate_model(config, checkpoint_path):

    # ---------- Load dataset ----------
    train, test = prepare_dataset(**config["data"])
    test = train['test'] # Ensure validation set is used for threshold tuning

    # ---------- Quantization ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["load_in_4bit"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=config["model"]["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
    )


    # ---------- Load tokenizer ----------
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_test = test.map(lambda x: tokenize_fn_val(x, tokenizer), batched=False)
    tokenized_test = tokenized_test.remove_columns(['labels'])
    tokenized_test.set_format(type="torch")#, columns=["input_ids", "attention_mask"])
    from torch.utils.data import DataLoader

    loader = DataLoader(tokenized_test, batch_size=1)#config['peft']['sft_config']['per_device_eval_batch_size'])
    preds = []
    labels = []
    map_dict = {"yes": 1, "no": 0}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["load_in_4bit"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
        )
    model = AutoModelForCausalLM.from_pretrained(config["model"]["model_name"], 
                                                 device_map=config["model"]["device_map"], 
                                                 quantization_config=bnb_config,
                                                 use_cache=False)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    scores_list = []
    label_ids_list = []
    yes_no_map = {5081: 1, 708: 0}

    with torch.inference_mode():
        for  ex in tqdm(loader, desc="Evaluating", total=len(loader)):
            # move inputs to the configured device (and keep dtypes correct)
            # inputs = tokenizer(ex['prompt'], add_special_tokens=False, return_tensors='pt')
            inputs = deepcopy(ex)
            inputs.pop('completion')
            inputs.pop('prompt')
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2,  # Only expect "yes" or "no" tokens
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                label_ids = torch.tensor((tokenizer(ex['completion'], add_special_tokens=False)['input_ids'][0])).unsqueeze(0)
                print(f"Predicted: {tokenizer.decode(outputs.scores[0].argmax())}, GT: {ex['completion'][0].strip("</s>")} ")

            pred = gather_yes_no_logprobs(outputs.scores[0], tokenizer)
            pred_scores = convert_scores_to_probs(pred)
            yes_no_label = label_ids[:,0].unsqueeze(1)
            yes_no_label = list(map(yes_no_map.get, yes_no_label.squeeze(1).tolist()))
            label_ids_list.append(yes_no_label)
            scores_list.append(pred_scores.item())
    all_scores = torch.tensor(scores_list)
    all_label_ids = torch.tensor(label_ids_list)
    thresh = find_best_threshold(all_scores.cpu(), all_label_ids.cpu().squeeze(1))
    print(f"Best threshold found: {thresh}")


# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", default=None)
    
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    evaluate_model(config, args.checkpoint)