from dataclasses import dataclass


@dataclass
class Music:
    """Class that represent all data about a music that will be used as the 
    input of the prompt generator class.
    """

    name: str
    clap_desc: str
    metadata: str = None
    instruction_id: int = 1
