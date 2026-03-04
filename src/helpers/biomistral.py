import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


from .base import HFInference



class BioMistralInference(HFInference):
    def __init__(self, peft_checkpoint=None, bnb_config=None):
        tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
        tokenizer.pad_token = tokenizer.eos_token
        
        if bnb_config is None:
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # "nf4" (normal float 4) gives better accuracy than "fp4"
            bnb_4bit_use_double_quant=True,  # optional, enables nested quantization for efficiency
            bnb_4bit_compute_dtype=torch.bfloat16  # preferred compute dtype; can also use torch.float16
            )
            
        model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B", quantization_config=bnb_config, 
                                          device_map={"":"cuda:0"}, low_cpu_mem_usage=True)
        if peft_checkpoint:
            model = PeftModel.from_pretrained(model, peft_checkpoint)
        model.config.pad_token_id = model.config.eos_token_id
        print("BioMistral model loaded.")
        self.model = model
        self.tokenizer = tokenizer

    def inference(self, messages, max_new_tokens=256):
        input_ = self.tokenizer(messages, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **input_,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    
    @torch.no_grad()
    def classify_yes_no(self, prompt):
        options = ["yes</s>", "no</s>"]
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        base_ids = enc["input_ids"]
        base_len = base_ids.size(1)

        self.model.eval()
        scores = []

        with torch.inference_mode():
            for opt in options:
                opt_ids = self.tokenizer(opt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(self.model.device)
                full_ids = torch.cat([base_ids, opt_ids], dim=1)

                out = self.model(full_ids, attention_mask=torch.ones_like(full_ids))
                logits = out.logits[:, base_len-1 : base_len-1 + opt_ids.size(1), :]
                logprobs = torch.log_softmax(logits, dim=-1)

                gather_idx = opt_ids.unsqueeze(-1)
                token_logprobs = logprobs.gather(2, gather_idx).squeeze(-1)

                scores.append(token_logprobs.sum().item())
        return options[scores.index(max(scores))].lower()
