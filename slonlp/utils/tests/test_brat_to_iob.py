import re
import string
from pathlib import Path

import pytest
import nltk
from nltk.tokenize import RegexpTokenizer

from slonlp.utils.brat_to_iob import read_spans_annotation
from slonlp.utils.brat_to_iob import build_iob, build_brat
from slonlp.utils.brat_to_iob import covert_brat_to_conll
from slonlp.utils.tokenization import tokenize_text
from slonlp.utils.tokenization import get_sentence_tokenizer, get_word_tokenizer


def test_build_iob():
    brat_dir = Path(__file__).parent / 'data'
    brat_doc_name = '34339291023600645023003_2'
    brat_text_path = (brat_dir / brat_doc_name).with_suffix('.txt')
    brat_ann_path = brat_text_path.with_suffix('.ann')

    annotation_src = read_spans_annotation(brat_ann_path)
    text = brat_text_path.open('rt').read()

    sentences = tokenize_text(get_sentence_tokenizer(), get_word_tokenizer(),
                              text)

    annotation_new = build_brat(build_iob(text, sentences, annotation_src),
                                text)

    for e1, e2 in zip(annotation_src, annotation_new):
        assert e1.label == e2.label, "Labels don't match"
        assert e1.text == e2.text, "Text don't match"


def test_covert_brat_to_conll(tmpdir):
    brat_dir = Path(__file__).parent / 'data'
    brat_doc_name = '34339291023600645023003_2'
    conll_dir = Path(str(tmpdir))
    covert_brat_to_conll(brat_dir, conll_dir, brat_doc_name)
    conll_path = (conll_dir / brat_doc_name).with_suffix('.txt')

    assert conll_path.is_file(), "Result file not found"

    text = (brat_dir / brat_doc_name).with_suffix('.txt').open('rt').read()
    annotation_new = build_brat((s.strip() for s in conll_path.open('rt')),
                                text)
    annotation_src = read_spans_annotation(
        (brat_dir / brat_doc_name).with_suffix('.ann'))

    for e1, e2 in zip(annotation_src, annotation_new):
        assert e1.label == e2.label, "Labels don't match"
        assert e1.text == e2.text, "Text don't match"
