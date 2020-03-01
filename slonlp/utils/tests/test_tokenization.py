from slonlp.utils.tokenization import tokenize_text, get_sentence_tokenizer, get_word_tokenizer


def test_tokenization():
    text = "У попа была собака. Он ее продал за 1.48 рублей."
    sents = tokenize_text(get_sentence_tokenizer(), get_word_tokenizer(), text)
    glued = [' '.join(text[s:e] for s, e in sent) for sent in sents]
    assert len(glued) == 2, "Sentence number doesn't match"
    assert glued[0] == 'У попа была собака .', "Sentence tokens don't match"
    assert glued[
        1] == 'Он ее продал за 1 . 48 рублей .', "Sentence tokens don't match"
