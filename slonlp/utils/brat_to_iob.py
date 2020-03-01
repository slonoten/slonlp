"""Convertation of brat text annotation format to IOB"""

from typing import List, Tuple, Iterable
from pathlib import Path
import nltk
from collections import namedtuple
from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer, PunktSentenceTokenizer

Entity = namedtuple('Entity', 'id label position text')
Position = namedtuple('Position', 'start end')


def read_spans_annotation(path: Path) -> List[Entity]:
    """Read text span annotations from "ann" file.

    :param path: path to "ann" file
    :type path: Path
    :return: list of annotated text spans
    :rtype: List[Entity]
    """
    entities = [
        Entity(id_, label, Position(int(start), int(end)), phrase)
        for id_, label, start, end, phrase in (
            (id_, *part.split(), phrase.rstrip())
            for id_, part, phrase in (s.split('\t') for s in path.open()
                                      if s.startswith('T')))
    ]
    return sorted(entities, key=lambda e: e.position.start)


def build_iob(text: str, sentences: Iterable[List[Tuple[int, int]]],
              annotation: List[Entity]) -> Iterable[str]:
    """Builds sequence of tokens with labels and position info from 
    text splited by sentences and tokens and annotated spans
    
    :param text: document text
    :type text: str
    :param sentences: list of sentence's token positions 
    :type sentences: Iterable[List[Tuple[int, int]]]
    :param annotation: annotated spans
    :type annotation: List[Entity]
    :yield: CONLL formated strings for document
    :rtype: Iterable[str]
    """
    entity_index = 0
    entity = annotation[entity_index] if annotation else None
    label = None
    entity_started = False
    for sent in sentences:
        for start, end in sent:
            if entity and start >= entity.position.end:
                entity_index += 1
                entity = annotation[entity_index] if len(
                    annotation) > entity_index else None
                entity_started = False
            if entity and start >= entity.position.start and end <= entity.position.end:
                if not entity_started and label == entity.label:
                    prefix = 'B'
                else:
                    prefix = 'I'
                label = entity.label
                tag = prefix + '-' + label
                entity_started = True
            else:
                label = 'O'
                tag = label
            yield '\t'.join(map(str, (text[start:end], tag, start, end)))
        yield ''


def iob_to_brat(conll: Iterable[str], text: str) -> Iterable[Entity]:
    """[summary]
    
    :param conll: sequence of strings in COLL format (token, label, start, end) in IOB format
    :type conll: Iterable[str]
    :param text: document text
    :type text: str
    :yield: annotated text spans
    :rtype: Iterable[Entity]
    """
    entity_id = 1
    entity = None
    for s in conll:
        if not s:
            continue
        _, label, start_str, end_str = s.split('\t')
        start, end = int(start_str), int(end_str)
        if entity:
            if label == 'O':
                yield entity
                entity = None
            elif label.startswith('B-') or label[2:] != entity.label:
                yield entity
                entity = Entity(f'T{entity_id}', label[2:],
                                Position(start, end), text[start:end])
                entity_id += 1
            else:
                assert label == 'I-' + entity.label
                entity = Entity(entity.id, entity.label,
                                Position(entity.position.start, end),
                                text[entity.position.start:end])
            continue
        if label == 'O':
            continue
        entity = Entity(f'T{entity_id}', label[2:], Position(start, end),
                        text[start:end])
        entity_id += 1
    if entity:
        yield entity
