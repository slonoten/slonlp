"""Text tokenization helpers"""

import string
import re
from typing import List, Tuple, Iterable

import nltk
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer
from nltk.tokenize.api import TokenizerI


def tokenize_text(sent_tokenizer: PunktSentenceTokenizer,
                  word_tokenizer: TokenizerI,
                  text: str) -> Iterable[List[Tuple[int, int]]]:
    """Splits text to sentences and sentences to tokens
    
    :param sent_tokenizer: sentence detector
    :type sent_tokenizer: PunktSentenceTokenizer
    :param word_tokenizer: word tokenizer
    :type word_tokenizer: RegexpTokenizer
    :param text: text to split
    :type text: str
    :yield: list of sentence tokens start and end positions
    :rtype: Iterable[List[Tuple[int, int]]]
    """
    paragraphs = text.split('\n')
    para_sents = list(sent_tokenizer.span_tokenize_sents(paragraphs))
    parargraph_start = 0
    for para_sents, para_text in zip(para_sents, paragraphs):
        for sent_start, sent_end in para_sents:
            sentence = []
            sent_text = text[parargraph_start + sent_start:parargraph_start +
                             sent_end]
            for token_start, token_end in word_tokenizer.span_tokenize(
                    sent_text):
                offset = parargraph_start + sent_start
                sentence.append((offset + token_start, offset + token_end))
            yield sentence
        parargraph_start += len(para_text) + 1


def get_sentence_tokenizer(lang: str = 'russian') -> PunktSentenceTokenizer:
    """Returns sentence detector
    
    :param lang: language, defaults to 'russian'
    :type lang: str, optional
    :return: sentence detector
    :rtype: PunktSentenceTokenizer
    """
    return nltk.data.load(f'tokenizers/punkt/{lang}.pickle')


def get_word_tokenizer() -> RegexpTokenizer:
    """Returns word tokenizer
    
    :return: word tokenizer
    :rtype: RegexpTokenizer
    """
    return RegexpTokenizer(f'\\w+|[{re.escape(string.punctuation)}]|\\S+')
