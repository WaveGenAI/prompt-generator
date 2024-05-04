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


PROMPT_LST = {
    1: _PROMPT_MUSIC_GEN.copy(),
    2: _PROMPT_TTS.copy(),
    3: _PROMPT_REMIX.copy(),

}


PROMPT_FORMAT_LYRICS = [
    """You have been given a text that is originally a song lyric. Your task is to reformat this text to clearly reflect its structure as a song, identifying and labeling each part of the song (e.g., verse, chorus, bridge) accordingly. A chorus is repeated and a verse change each time. Number the verse. Use brackets to label each part at the beginning, such as [Verse1], [Chorus], [Bridge], etc. Separate each line clearly, and ensure that stanzas are properly distinguished from each other. Preserve rhyme scheme, and any repetitive elements exactly as they appear in the original lyric. Here is the text:

{LYRICS}

Reformat and label the lyric to highlight the intended flow of the song. Make sure each part of the song is easily identifiable and formatted for readability."""
]

# the instruction for the phi3 model
INSTRUCT_PHI3 = {"role": "system",
                 "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."}
