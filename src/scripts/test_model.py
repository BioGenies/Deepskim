from argparse import ArgumentParser
import yaml 
from collections import namedtuple
import csv
import numpy as np
from copy import deepcopy
import re

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
from utils.evaluation import gather_yes_no_logprobs, convert_scores_to_probs, convert_probs_to_labels, percent_to_review_for_recall

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


def evaluate_model(config, checkpoint_path=None, save_false_preds=False):

    # ---------- Load dataset ----------
    train, test = prepare_dataset(**config["data"])
    # test = train['test'] # Check that the training and evaluation are implemented okay

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

    # model = BioMistralInference(peft_checkpoint=checkpoint_path, bnb_config=bnb_config)

    # device = next(model.model.parameters()).device  # works for QLoRA / regular models

    loader = DataLoader(tokenized_test, batch_size=1)#config['peft']['sft_config']['per_device_eval_batch_size'])
    preds = []
    labels = []
    scores = []
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
    eval_obj = namedtuple("EvalObj", ["predictions", "label_ids"])
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

            # Decode only the generated tokens (skip input)
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            label_ids = torch.tensor((tokenizer(ex['completion'], add_special_tokens=False)['input_ids'][0]))[0].unsqueeze(0).unsqueeze(0)
            pred = gather_yes_no_logprobs(outputs.scores[0], tokenizer)
            pred_scores = convert_scores_to_probs(pred)
            pred_labels = convert_probs_to_labels(pred_scores, tokenizer, threshold=config['evaluation']['decision_threshold'])
            scores.append(pred_scores.item())
            preds.append(pred_labels.item())
            labels.append(map_dict[ex['completion'][0][:-4]])
            # eval_preds = eval_obj(predictions=pred.unsqueeze(0), label_ids=label_ids)
            # compute_result = idx == len(loader) - 1
            # metrics = compute_metrics(eval_preds, tokenizer, compute_result=compute_result, shift=False)
            # print("PROMPT :", ex["prompt"])
            # print("TARGET :", ex["completion"])
            # print("PRED   :", generated_text)
            # print()
            # if metrics is not None:
            #     print(metrics)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    percent_to_review = percent_to_review_for_recall(list(zip(preds, scores)), labels, recall_target=0.95)
    print(f"Final Evaluation Metrics at threshold {config['evaluation']['decision_threshold']}:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Percent to review for 95% recall: {percent_to_review:.2f}%")

    if save_false_preds:
        strong_fps = []
        strong_fns = []
        for idx in range(len(labels)):
            if labels[idx] == 1 and scores[idx] < 0.1:
                strong_fns.append(idx)
            elif labels[idx] == 0 and scores[idx] > 0.9:
                strong_fps.append(idx)

        header_row = ["Title", "Abstract", "Journal", "Referencess", "FP/FN"]
        with open("strong_false_predictions.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header_row)
            for idx, data in enumerate(loader.dataset):
                if idx in strong_fns or idx in strong_fps:
                    text = data['prompt']
                    title = re.search("Title: (.*)\n", text)
                    abstract = re.search("Abstract: (.*)\n", text)
                    journal = re.search("Journal: (.*)\n", text)
                    refs = re.search("References: (.*)\n", text)
                    false_type = "FN" if idx in strong_fns else "FP"
                    row = [title.group(1), abstract.group(1), journal.group(1), refs.group(1) if refs else '', false_type]
                    writer.writerow(row)



# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_false_preds", action='store_true', help="Whether to save false predictions for error analysis")
    
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    evaluate_model(config, args.checkpoint, save_false_preds=args.save_false_preds)