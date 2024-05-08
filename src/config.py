_PROMPT_MUSIC_GEN = [
    "Build a prompt of 1~2 lines that will be put in a music generation model to reproduce the music. \
        Split each element with a ',' and insert only element for which the description is most likely correct. \
            The most reliable information are, in order, the title, the metadatas and after the music description. \
                You will not put the title directly in the prompt, but you can use the title to obtain some element and put them in the prompt. \
                    The music \"{NAME}\" metadatas are: {METADATA} and the full no-accurate description of the music for each slice of 10 seconds is: {CLAPS}.",
]


_PROMPT_TTS = [
    "hello world this is a test"
]

_PROMPT_REMIX = [
    "hello world this is a test"
]

_PROMPT_LYRICS = [
    "hello world this is a test"
]

PROMPT_LST = {
    1: _PROMPT_MUSIC_GEN.copy(),
    2: _PROMPT_TTS.copy(),
    3: _PROMPT_REMIX.copy(),
    4: _PROMPT_LYRICS.copy(),
}

# the instruction for the phi3 model
INSTRUCT_PHI3 = {"role": "system",
                 "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."}
