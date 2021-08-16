from trec_cds.data.utils import Gender


class Topic:
    gender: Gender
    age: int

    def __init__(self, number, text):
        self.number = number
        self.text = text


# from collections import namedtuple
# Topic = namedtuple('Topic', 'number text gender text')
