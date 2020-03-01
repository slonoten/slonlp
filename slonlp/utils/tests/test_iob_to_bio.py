import sys

from slonlp.utils.bilou_to_iob import iob_to_bio


def test_serial():
    iob = ['O', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
    bio = ['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
    assert iob_to_bio(iob) == bio


def test_sequence_then_single():
    iob = ['O', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O']
    bio = ['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O']
    assert iob_to_bio(iob) == bio


def test_sequence():
    iob = ['O', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O']
    bio = ['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'O']
    assert iob_to_bio(iob) == bio


def test_single_then_sequence():
    iob = ['O', 'O', 'I-LOC', 'B-LOC', 'I-LOC']
    bio = ['O', 'O', 'B-LOC', 'B-LOC', 'I-LOC']
    assert iob_to_bio(iob) == bio, str(iob_to_bio(iob))