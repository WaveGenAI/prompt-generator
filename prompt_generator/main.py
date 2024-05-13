import random
from typing import List

from .config import PROMPT_LST, PROMPT_FORMAT_LYRICS
from .data.lyrics import Lyrics
from .data.music import Music
from .data.prompt import Prompt
from .model.phi3 import Phi3Model


class PromptGenerator:
    def __init__(self, batch_size: int = 1):
        self.model = Phi3Model(batch_size=batch_size)

    def generate_formated_lyrics(self, lyrics: List[Lyrics] | Lyrics) -> List[Lyrics]:

        lst_lyrics = []

        for lyric in (lyrics if isinstance(lyrics, list) else [lyrics]):
            lst_lyrics.append(Lyrics(PROMPT_FORMAT_LYRICS[0].replace('{LYRICS}', lyric.content)))    # just a trick

        return self.model.generate_response(lst_lyrics)

    def generate_prompt_from_music(self, musics: List[Music] | Music) -> List[str]:
        """Method to generate prompt from music data.

        :param musics: the list of music data that will be used to generate the prompt
        :type musics: List[Music]
        :raises ValueError: if the music does not have a corresponding prompt list
        :return: the generated prompt that corresponds to the music data
        :rtype: List[str]
        """

        lst_prompts = []

        # iterate over the music data
        for music in (musics if isinstance(musics, list) else [musics]):
            if music.instruction_id not in PROMPT_LST:
                raise ValueError(
                    f"Music with instruction id {music.instruction_id} does not have a corresponding prompt list."
                )

            # get the prompt list based on the instruction id
            prompts = random.choice(PROMPT_LST[music.instruction_id])

            # filter clap description
            music.clap_desc = music.clap_desc.replace('The low quality recording features ', '').replace(
                'the recording is noisy', '').replace('The audio quality is poor', '').replace('in mono', '')

            # replace the placeholder with the actual music description
            prompts = prompts.replace("{CLAPS}", music.clap_desc)
            prompts = prompts.replace("{METADATA}", "")   # music.metadata)
            prompts = prompts.replace("{NAME}", music.name)

            lst_prompts.append(Prompt(prompts))

        return self.model.generate_response(lst_prompts)

    def generate_lyrics_from_music(self, musics: List[Music] | Music) -> List[str]:
        
        lst_prompts = []
        for music in (musics if isinstance(musics, list) else [musics]):
            if music.instruction_id not in PROMPT_LST:
                raise ValueError(
                    f"Music with instruction id {music.instruction_id} does not have a corresponding prompt list."
                )

            prompts = random.choice(PROMPT_LST[music.instruction_id])
            prompts = prompts.replace("{CLAPS}", music.clap_desc)
            prompts = prompts.replace("{METADATA}", music.metadata)
            prompts = prompts.replace("{NAME}", music.name)

            lst_prompts.append(Prompt(prompts))
            
        return self.model.generate_response(lst_prompts)
