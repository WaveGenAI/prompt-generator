_PROMPT_MUSIC_GEN = [
    "Describe the music {NAME}. The music metadata are {METADATA} and the full description is {CLAPS}.",
]

_PROMPT_TTS = [
    "hello world this is a test"
]

_PROMPT_REMIX = [
    "hello world this is a test"
]

PROMPT_LST = {
    1: _PROMPT_MUSIC_GEN.copy(),
    2: _PROMPT_TTS.copy(),
    3: _PROMPT_REMIX.copy
}

# the instruction for the phi3 model
INSTRUCT_PHI3 = {"role": "system",
                 "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."}
