import transformers
import torch
from transformers import AutoTokenizer

from .base import HFInference

model_id = "aaditya/OpenBioLLM-Llama3-8B"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

llama3_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False)
pipeline.tokenizer.chat_template = llama3_tok.chat_template 

class OpenBioLLMInference(HFInference):
    def __init__(self, pipeline=pipeline):
        self.pipepline = pipeline

    def inference(self, messages, max_new_tokens=256):
        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
        )
        return (outputs[0]["generated_text"][len(prompt):])
