from dataclasses import dataclass
from typing import Union, List

from trec_cds.data.utils import Gender


@dataclass
class Topic:
    """dataclass containing topic data.
    Number and text are loaded directly from the xml file.
    Other variables are parsed and created based on the topic's text."""
    number: int
    text: str

    gender: Gender = Gender.unknown
    age: Union[int, float, None] = None
    healthy: bool = True

    # text which was preprocessed and is already tokenized
    text_preprocessed: Union[None, List[str]] = None
