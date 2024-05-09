from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import INSTRUCT_PHI3
from ..data.lyrics import Lyrics
from ..data.prompt import Prompt


class Phi3Model:
    """Interface to use the Phi-3 model.
    """

    def __init__(self, batch_size: int = 7):  # for 8 batch sizes
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.batch_size = batch_size

    def generate_response(self, prompts: List[Prompt | Lyrics] | Prompt | Lyrics, ) -> List[str]:
        """Method to generate response from the model.

        :param prompts: The prompts that will be used to generate the response
        :type prompts: List[Prompt | Lyrics] | Prompt | Lyrics
        :return: the generated response that corresponds to the prompt
        :rtype: List[str]
        """

        msgs = []

        # convert the prompt to the format that the model can understand
        if isinstance(prompts, list):
            for prompt in prompts:
                msgs.append(
                    self.tokenizer.apply_chat_template(
                        [INSTRUCT_PHI3, {"role": "user", "content": prompt.content}], add_generation_prompt=True, return_tensors="pt", tokenize=False)
                )
        else:
            msgs.append(self.tokenizer.apply_chat_template(
                [INSTRUCT_PHI3, {"role": "user", "content": prompts.content}], add_generation_prompt=True, return_tensors="pt", tokenize=False)
            )

        inputs = self.tokenizer(msgs, return_tensors="pt",
                                padding=True).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=512,
                                  do_sample=True, temperature=0.5, top_k=20, top_p=0.95).tolist()

        # decode the output and remove the input prompt
        output = []
        for idx, out in enumerate(self.tokenizer.batch_decode(out, skip_special_tokens=True)):
            cleaned_output = out.replace(prompts[idx].content if isinstance(
                prompts, list) else prompts.content, "").strip()
            prompts[idx].content = cleaned_output
            output.append(prompts[idx])

        return output
