from typing import List

from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

from ..config import INSTRUCT_PHI3
from ..data.lyrics import Lyrics
from ..data.prompt import Prompt


class Phi3Model:
    """Interface to use the Phi-3 model.
    """

    def __init__(self, batch_size: int = 7):  # for 8 batch sizes
        self.llm = Llama.from_pretrained(
            repo_id="pjh64/Phi-3-mini-128K-Instruct.gguf",
            filename="*q4_k_m.gguf",
            flash_attn=True,
            n_gpu_layers=-1,
            n_ctx=6000,
            use_mlock=False,
            verbose=True,
            draft_model=LlamaPromptLookupDecoding(
                max_ngram_size=3, num_pred_tokens=5)  # boost?
        )
        self.batch_size = batch_size

    def generate_response(self, prompts: List[Prompt | Lyrics] | Prompt | Lyrics, ) -> List[Prompt | Lyrics]:
        """Method to generate response from the model.

        :param prompts: The prompts that will be used to generate the response
        :type prompts: List[Prompt | Lyrics] | Prompt | Lyrics
        :return: the generated response that corresponds to the prompt
        :rtype: List[Prompt | Lyrics]
        """

        msgs = []
        if isinstance(prompts, list):
            for prompt in prompts:
                prompt.content = self.llm.create_chat_completion(
                    [INSTRUCT_PHI3, {"role": "user", "content": prompt.content}])["choices"][0]["message"]["content"]
                msgs.append(prompt)
        else:
            msgs.append(self.llm.create_chat_completion([INSTRUCT_PHI3, {
                        "role": "user", "content": prompts.content}])["choices"][0]["message"]["content"])
        return msgs
