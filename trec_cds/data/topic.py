from dataclasses import dataclass
from typing import Union, List

from trec_cds.data.utils import Gender


@dataclass
class Topic:
    number: int
    text: str

    gender: Gender = Gender.unknown
    age: Union[int, float, None] = None
    healthy: bool = True

    # text which was preprocessed and is already tokenized
    text_preprocessed: Union[None, List[str]] = None
