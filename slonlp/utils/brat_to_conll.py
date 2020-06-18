"""Convertation of brat text annotation format to IOB"""

from typing import List, Tuple, Iterable
from pathlib import Path
from collections import namedtuple
from itertools import chain
import argparse
import logging

import tqdm

from slonlp.utils.tokenization import (
    tokenize_text,
    get_sentence_tokenizer,
    get_word_tokenizer,
)

Entity = namedtuple("Entity", "id label position text")
Position = namedtuple("Position", "start end")


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
            (id_, *part.split(), " ".join(phrase).rstrip())
            for id_, part, *phrase in (
                s.split("\t") for s in path.open() if s.startswith("T")
            )
        )
    ]
    sorted_entities = sorted(entities, key=lambda e: e.position.start)
    entitiy_pairs = zip(sorted_entities, sorted_entities[1:])
    nested_pairs = list(
        filter(lambda p: p[0].position.end > p[1].position.start, entitiy_pairs)
    )
    if nested_pairs:
        printable = ", ".join(f"{l.id} {r.id}" for l, r in nested_pairs)
        logging.warn(f'Nested annotations {printable} found in file "{path}"')
    return sorted_entities


def build_iob(
    text: str, sentences: Iterable[List[Tuple[int, int]]], annotation: List[Entity]
) -> Iterable[str]:
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
                entity = (
                    annotation[entity_index] if len(annotation) > entity_index else None
                )
                entity_started = False
            if entity and start >= entity.position.start and end <= entity.position.end:
                if not entity_started and label == entity.label:
                    prefix = "B"
                else:
                    prefix = "I"
                label = entity.label
                tag = prefix + "-" + label
                entity_started = True
            else:
                label = "O"
                tag = label
            yield "\t".join(map(str, (text[start:end], tag, start, end)))
        yield ""


def build_brat(conll: Iterable[str], text: str) -> Iterable[Entity]:
    """Builds BRAT text spans sequence from lines of COLL format
    
    :param conll: sequence of strings in COLL format in IOB format 
                  (token, label, start, end) 
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
        print(s)
        _, label, start_str, end_str = s.split("\t")
        start, end = int(start_str), int(end_str)
        if entity:
            if label == "O":
                yield entity
                entity = None
            elif label.startswith("B-") or label[2:] != entity.label:
                yield entity
                entity = Entity(
                    f"T{entity_id}", label[2:], Position(start, end), text[start:end]
                )
                entity_id += 1
            else:
                assert label == "I-" + entity.label
                entity = Entity(
                    entity.id,
                    entity.label,
                    Position(entity.position.start, end),
                    text[entity.position.start : end],
                )
            continue
        if label == "O":
            continue
        entity = Entity(
            f"T{entity_id}", label[2:], Position(start, end), text[start:end]
        )
        entity_id += 1
    if entity:
        yield entity


def covert_brat_to_conll(brat_dir: Path, doc_name: str) -> Iterable[str]:
    """Converts BRAT's document annotation to CONLL file"
    
    :param brat_dir: path to directory with brat annotations
    :type brat_dir: Path
    :param doc_name: name of annotated document
    :type doc_name: str
    """
    brat_text_path = (brat_dir / doc_name).with_suffix(".txt")
    brat_ann_path = brat_text_path.with_suffix(".ann")

    spans = read_spans_annotation(brat_ann_path)
    text = brat_text_path.open("rt").read()

    sentences = tokenize_text(get_sentence_tokenizer(), get_word_tokenizer(), text)

    return build_iob(text, sentences, spans)


def convert_to_multiple_files(input_dir: Path, output_path: Path):
    for ann_path in tqdm.tqdm(list(input_dir.glob("*.ann"))):
        doc_name = ann_path.stem
        conll = covert_brat_to_conll(input_dir, doc_name)
        conll_path = (output_path / doc_name).with_suffix(".txt")
        conll_path.open("wt").writelines(s + "\n" for s in conll)


def convert_to_single_file(input_dir: Path, output_path: Path):
    mode = "wt"
    for ann_path in tqdm.tqdm(list(input_dir.glob("*.ann"))):
        doc_name = ann_path.stem
        conll = covert_brat_to_conll(input_dir, doc_name)
        doc_start = "\t".join(("-DOCSTART-", doc_name, "-X-", "-X-"))
        output_path.open(mode).writelines(
            s + "\n" for s in chain((doc_start, ""), conll)
        )
        mode = "at"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python brat_to_conll -i <input dir> -o <output path>"
    )
    parser.add_argument("--input_dir", "-i", required=True)
    parser.add_argument("--output_path", "-o", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    if output_path.is_dir():
        convert_to_multiple_files(input_dir, output_path)
    elif output_path.parent.is_dir():
        convert_to_single_file(input_dir, output_path)
    else:
        print(
            "Error: output path should be path to existing directory or to file in existing directory"
        )
