from typing import List

from .data.music import Music
from .data.prompt import Prompt

from .config import PROMPT_LST


class PromptGenerator:
    def generate_prompt_from_music(self, musics: List[Music]) -> List[Prompt]:
        return []
