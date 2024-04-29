from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Phi3Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate_response(self, messages):
        msg = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=False)
        inputs = self.tokenizer((msg, msg), return_tensors="pt", padding=True).to(self.model.device)
        output_sequences = self.model.generate(**inputs, max_new_tokens=20)
        return self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
