from typing import Union

from trec_cds.data.utils import Gender


class Topic:
    gender: Gender
    age: Union[int, float, None]
    healthy: bool

    def __init__(self, number: int, text: str):
        self.number: int = number
        self.text: str = text


# from collections import namedtuple
# Topic = namedtuple('Topic', 'number text gender text')
