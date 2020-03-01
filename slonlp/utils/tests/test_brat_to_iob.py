import re
import string
from pathlib import Path

import nltk
from nltk.tokenize import RegexpTokenizer

from slonlp.utils.brat_to_iob import read_spans_annotation, build_iob, iob_to_brat
from slonlp.utils.tokenization import tokenize_text


def test_brat_to_iob():
    brat_dir = Path(__file__).parent / 'data'
    brat_doc_name = '34339291023600645023003_2'
    brat_text_path = (brat_dir / brat_doc_name).with_suffix('.txt')
    brat_ann_path = brat_text_path.with_suffix('.ann')

    annotation = read_spans_annotation(brat_ann_path)
    text = brat_text_path.open('rt').read()

    sent_detector = nltk.data.load('tokenizers/punkt/russian.pickle')
    sentences = list(
        tokenize_text(
            sent_detector,
            RegexpTokenizer(f'\\w+|[{re.escape(string.punctuation)}]|\\S+'),
            text))

    annotation_new = list(
        iob_to_brat(build_iob(text, sentences, annotation), text))

    for e1, e2 in zip(annotation, annotation_new):
        assert e1.label == e2.label, "Labels don't match"
        assert e1.text == e2.text, "Text don't match"
