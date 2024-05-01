from typing import List

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import INSTRUCT_PHI3
from ..data.prompt import Prompt


class Phi3Model:
    """Interface to use the Phi-3 model.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    def generate_response(self, prompts: List[Prompt] | Prompt, ) -> List[Prompt]:
        """Method to generate response from the model.

        :param prompts: the prompts that will be used to generate the response
        :type prompts: List[Prompt] | Prompt
        :return: the generated response that corresponds to the prompt
        :rtype: List[Prompt]
        """

        msgs = []
        batch_size = 7     # for 8 batch sizes

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
        output_sequences = None
        msgs_batch = np.array_split(np.array(msgs), len(msgs) // batch_size + 1)
        for msgs in msgs_batch:
            if msgs.size == 0:
                continue
            inputs = self.tokenizer(
                msgs.tolist(), return_tensors="pt", padding=True).to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=20, top_p=0.90).tolist()
            if output_sequences is None:
                output_sequences = out
            else:
                for micro_batch in out:
                    output_sequences.append(micro_batch)

        # decode the output and remove the input prompt
        output = []
        for idx, out in enumerate(self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)):
            if isinstance(prompts, list):
                output.append(Prompt(out.replace(
                    prompts[idx].content, "").strip()))

                continue

            output.append(Prompt(out.replace(prompts.content, "").strip()))

        return output
