from typing import List

import languagemodels as lm
import numpy as np
from transformers import AutoTokenizer

from ..config import INSTRUCT_PHI3
from ..data.lyrics import Lyrics
from ..data.prompt import Prompt


class Phi3Model:
    """Interface to use the Phi-3 model.
    """

    def __init__(self, batch_size: int = 7): # for 8 batch sizes
        self.tokenizer = AutoTokenizer.from_pretrained(
            "leliuga/Phi-3-mini-128k-instruct-bnb-4bit", trust_remote_code=True)
        lm.config['instruct_model'] = 'Phi-3-mini-4k-instruct'
        lm.config["device"] = "auto"
        self.batch_size = batch_size

    def generate_response(self, prompts: List[Prompt | Lyrics] | Prompt | Lyrics, ) -> List[Prompt | Lyrics]:
        """Method to generate response from the model.

        :param prompts: The prompts that will be used to generate the response
        :type prompts: List[Prompt | Lyrics] | Prompt | Lyrics
        :return: the generated response that corresponds to the prompt
        :rtype: List[Prompt | Lyrics]
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
        output_sequences = []
        msgs_batch = np.array_split(np.array(msgs), len(msgs) // self.batch_size + 1)
        i = 0
        for msgs in msgs_batch:
            if msgs.size == 0:
                continue

            out = lm.generate(msgs.tolist(), max_tokens=300, temperature=0.2, topk=10,)
            if output_sequences is None:
                output_sequences = out
            else:
                for micro_batch in out:
                    prompts[i].content = micro_batch
                    output_sequences.append(prompts[i])
                    i += 1

        return output_sequences
