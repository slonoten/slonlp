"""Conll labels convertation"""

from typing import List, Tuple
from pathlib import Path
import argparse
import json


def bilou_to_iob(labels: List[str]) -> List[str]:
    """Converts label sequence from BILOU to IOB format

    :param labels: label sequence in BILOU format
    :type labels: List[str]
    :return: label sequence in IOB format
    :rtype: List[str]
    """
    result = []
    last = ''
    for label in labels:
        if label == 'O':
            result.append(label)
            last = label
            continue

        prefix, cls = label.split('-')
        new_prefix = prefix
        if prefix in ('L', 'I'):
            new_prefix = 'I'
        elif prefix == 'U':
            if last in ('L-' + cls, 'U-' + cls):
                new_prefix = 'B'
            else:
                new_prefix = 'I'
        elif prefix == 'B':
            if last not in ('L-' + cls, 'U-' + cls):
                new_prefix = 'I'
        last = label
        result.append(new_prefix + '-' + cls)
    return result


def iob_to_bio(labels: List[str]) -> List[str]:
    """Converts label sequence from IOB to BIO format
    
    :param labels: label sequence in IOB format
    :type labels: List[str]
    :return: label sequence in BIO format
    :rtype: List[str]
    """
    result = []
    last = ''
    for label in labels:
        if label.startswith('I-') and last not in (label, 'B-' + label[2:]):
            result.append('B-' + label[2:])
        else:
            result.append(label)
        last = label
    return result


def load_conll(src_path: str) -> List[Tuple[List[str], List[str]]]:
    sentences = []
    with open(src_path, 'rt') as file:
        tokens: List[str] = []
        labels: List[str] = []
        for s in file:
            if s.startswith('-DOCSTART-'):
                continue
            if not s.strip():
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            token, *items = s.split()
            tokens.append(token)
            labels.append(items[-1])
        if tokens:
            sentences.append((tokens, labels))
    return sentences


def load_predictions(src_path: str) -> List[Tuple[List[str], List[str]]]:
    predictions = []
    with open(src_path, 'rt') as file:
        for sent in file:
            sentence_preds = json.loads(sent)
            predictions.append(
                (sentence_preds['words'], sentence_preds['tags']))
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "python bilou_to_iob -p <input json> -e <input conll> -o <output conll>"
    )
    parser.add_argument('--predict', '-p', required=True)
    parser.add_argument('--etalon', '-e', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--bio', '-b', action='store_true')
    args = parser.parse_args()
    pred = load_predictions(args.predict)
    label_postprocessing = iob_to_bio if args.bio else lambda x: x
    iob_pred = map(lambda s: (s[0], label_postprocessing(bilou_to_iob(s[1]))),
                   pred)
    etalon = load_conll(args.etalon)
    iob_etalon = map(lambda s: (s[0], label_postprocessing(s[1])), etalon)
    with open(args.output, 'wt') as out_file:
        for (tp, lp), (tg, lg) in zip(iob_pred, iob_etalon):
            assert tp == tg
            out_file.writelines([' '.join(p) + '\n' for p in zip(tp, lg, lp)])
            out_file.write('\n')
