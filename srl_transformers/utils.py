from collections import defaultdict
import pathlib
from typing import Dict


def load_role_frame(filename: pathlib.Path) -> Dict:
    """
    Open a dictionary from file, in the format (lemma,frame) -> [roles]
    :param filename: file to read.
    :return: a dictionary.
    """
    dictionary = defaultdict(list)
    with open(filename) as file:
        for l in file:
            lemma, frame, *v = l.split()
            dictionary[(lemma, frame)] += v
    return dictionary


def load_lemma_frame(filename: pathlib.Path) -> Dict:
    """
    Open a dictionary from file, in the format lemma -> [frames]
    :param filename: file to read.
    :return: a dictionary.
    """
    dictionary = defaultdict(list)
    with open(filename) as file:
        for l in file:
            k, *v = l.split()
            dictionary[k] += v
    return dictionary
