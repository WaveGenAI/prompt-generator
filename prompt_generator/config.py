_PROMPT_MUSIC_GEN = [
    """
    You are an expert in creating prompts for AI models for music generation. Your task is to write a concise prompt of 10-20 words to describe a piece of music based on the provided information.

    The name of the music piece is:
    <name>
    {NAME}
    </name>

    The metadata of the music piece is:
    <metadata>
    {METADATA}
    </metadata>

    A no-accurate description of the music for each 10-second slice is provided below:
    <claps>
    {CLAPS}
    </claps>

    <thinkingstep>
    Carefully analyze the name, metadata, and the no-accurate description of the music piece. Consider the key elements, such as the genre, mood, instruments, and the progression of the music over time. Identify the most important aspects that capture the essence of the music piece.
    </thinkingstep>

    Based on your analysis, write a concise and descriptive prompt of 10-20 words that encapsulates the core characteristics of the music piece. Output your prompt inside <prompt> tags.
    </thinkingstep>
    """.strip(),
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
    3: _PROMPT_REMIX.copy
}


PROMPT_FORMAT_LYRICS = [
    """
    Here is the text of a song lyric: 
    <lyrics>
    {LYRICS}
    </lyrics>

    Your task is to reformat this lyric to clearly reflect the structure and flow of the song. To do this:

    - Identify each distinct part of the song (verse, chorus, bridge, etc.) and label it using brackets at the beginning, like [Verse], [Chorus], [Bridge]
    - Number each verse sequentially ([Verse 1], [Verse 2], etc.)
    - Make sure stanzas are clearly distinguished from each other with a blank line in between
    - Keep the original rhyme scheme and any phrases exactly as they appear in the source lyric

    Write out the full song lyric with this new labeling and formatting. The goal is to make the intended flow and structure of the song obvious and easily readable.
    """,
]

# the instruction for the phi3 model
INSTRUCT_PHI3 = {"role": "system",
                 "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."}
