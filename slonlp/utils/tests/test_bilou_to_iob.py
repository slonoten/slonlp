import sys

from slonlp.utils.bilou_to_iob import bilou_to_iob


def test_serial():
    bilou = ['O', 'O', 'B-LOC', 'I-LOC', 'L-LOC', 'B-LOC', 'L-LOC']
    iob = ['O', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
    assert bilou_to_iob(bilou) == iob


def test_sequence_then_single():
    bilou = ['O', 'O', 'B-LOC', 'I-LOC', 'L-LOC', 'U-LOC', 'O']
    iob = ['O', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O']
    assert bilou_to_iob(bilou) == iob, bilou_to_iob(bilou)


def test_sequence():
    bilou = ['O', 'O', 'B-LOC', 'I-LOC', 'L-LOC', 'O', 'O']
    iob = ['O', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O']
    assert bilou_to_iob(bilou) == iob


def test_single_then_sequence():
    bilou = ['O', 'O', 'U-LOC', 'B-LOC', 'L-LOC']
    iob = ['O', 'O', 'I-LOC', 'B-LOC', 'I-LOC']
    assert bilou_to_iob(bilou) == iob
