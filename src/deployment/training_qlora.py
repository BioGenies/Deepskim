import math
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import DatasetDict, Dataset, concatenate_datasets

from data.dataset_qlora import prepare_dataset
from training.qlora import QLora


# def _tokenize(dataset, tokenizer):
#     if isinstance(dataset, DatasetDict):
#         dataset = DatasetDict({k: _tokenize(v, tokenizer) for k, v in dataset.items()})
#         return dataset
#     else:
#         tokenized_dataset = []
#         # input_idss = []
#         # completion_masks = []
#         # prompts = []
#         # completions = []
#         for item in dataset:
#             prompt_tokens = tokenizer(item['prompt'])['input_ids']
#             completion_tokens = tokenizer(item['completion'], add_special_tokens=False)['input_ids']
#             input_ids = prompt_tokens + completion_tokens
#             completion_mask = [0]*len(prompt_tokens) + [1]*len(completion_tokens)
#             tokenized_dataset.append({"input_ids": input_ids, "completion_mask": completion_mask, "prompt": item['prompt'], "completion": item['completion']})
#         return Dataset.from_list(tokenized_dataset)


def oversample_yes(dataset, positive_ratio=0.5):
    yes_ds = dataset.filter(lambda x: x["completion"].startswith("yes"))
    no_ds = dataset.filter(lambda x: x["completion"].startswith("no"))

    n_yes = len(yes_ds)
    n_no = len(no_ds)
    total = n_yes + n_no

    if total == 0:
        return dataset

    r = float(positive_ratio)
    if not (0.0 <= r < 1.0):
        raise ValueError(
            "positive_ratio must be in [0.0, 1.0) when only upsampling is allowed"
        )

    current_frac = n_yes / total if total > 0 else 0.0

    # If desired ratio is already met or would require undersampling, do nothing
    if r <= current_frac:
        return dataset

    # Need to upsample the positive (yes) class to achieve r in the final set.
    if n_yes == 0:
        raise ValueError(
            "Cannot upsample positives: no positive examples present in dataset"
        )

    # Solve for x: (n_yes + x) / (total + x) = r  =>  x = (r*total - n_yes) / (1 - r)
    numerator = r * total - n_yes
    denom = 1.0 - r
    to_add = math.ceil(numerator / denom) if numerator > 0 else 0

    target_yes = n_yes + to_add

    # Build upsampled positives by repeating yes_ds as needed
    full_repeats = target_yes // n_yes
    remainder = target_yes % n_yes
    parts = [yes_ds] * full_repeats
    if remainder:
        parts.append(yes_ds.shuffle().select(range(remainder)))
    yes_adjusted = concatenate_datasets(parts)

    balanced = concatenate_datasets([yes_adjusted, no_ds]).shuffle()
    return balanced


def tokenize_fn_train(example, tokenizer, max_length=4096):
    prompt = example["prompt"]
    completion = example["completion"]

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids

    # Create labels (mask out the prompt, keep completion including EOS)
    labels = [-100] * len(prompt_ids) + completion_ids

    if completion_ids[-1] == tokenizer.eos_token_id:
        labels[-1] = -100  # Mask out EOS token - not relevant for loss
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def tokenize_fn_val(example, tokenizer):
    prompt = example["prompt"]

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if prompt_ids[-1] == tokenizer.eos_token_id:
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


def train_model(config):

    set_seed(config.get("seed", 42))

    # setup device
    if config.get("device"):
        device = torch.device(config["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    run_name = (
        f"{config['run_name']}_model_{config['model']['model_name']}_lr{config['peft']['sft_config']['learning_rate']}"
        f"_peft_"
    )

    # ------- Load dataset -------
    train_val, _ = prepare_dataset(**config["data"])

    dtype_str = config["model"]["bnb_4bit_compute_dtype"]  # "bfloat16"
    compute_dtype = getattr(torch, dtype_str)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["load_in_4bit"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name"],
        device_map=config["model"]["device_map"],
        quantization_config=bnb_config,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = train_val["train"].map(
        lambda x: tokenize_fn_train(x, tokenizer), batched=False
    )
    # train_dataset = oversample_yes(train_dataset, positive_ratio=config["data"]["positive_ratio"])
    val_dataset = train_val["test"].map(
        lambda x: tokenize_fn_train(x, tokenizer), batched=False
    )

    # val_dataset = train_val['test'].map(lambda x: tokenize_fn_val(x, tokenizer), batched=False, remove_columns=['prompt', 'completion'])
    print(f"Model {config['model']['model_name']} loaded.")
    config["peft"]["sft_config"]["run_name"] = run_name
    qlora = QLora(
        model=model,
        tokenizer=tokenizer,
        lora_config=config["peft"]["lora_config"],
        sft_config=config["peft"]["sft_config"],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        device=device,
        positive_ratio=config["data"].get("positive_ratio", 0.3),
        label_smoothing=config.get("label_smoothing", 0.03),
        reason_weights=config["data"].get("reason_weights", None),
        continue_from=config["model"]["continue_from"],
    )

    # ------- Build & train -------
    qlora.train_model()
