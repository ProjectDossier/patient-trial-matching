from typing import Union, List

from trec_cds.data.utils import Gender


class Topic:
    gender: Gender
    age: Union[int, float, None]
    healthy: bool

    text_preprocessed: List[str]  # text which was preprocessed and is already tokenized

    def __init__(self, number: int, text: str):
        self.number: int = number
        self.text: str = text


# from collections import namedtuple
# Topic = namedtuple('Topic', 'number text gender text')
