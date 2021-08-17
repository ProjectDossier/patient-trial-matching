from trec_cds.data.utils import Gender
from datetime import datetime


class Topic:
    gender: Gender
    age: datetime
    healthy: bool

    def __init__(self, number, text):
        self.number = number
        self.text = text


# from collections import namedtuple
# Topic = namedtuple('Topic', 'number text gender text')
