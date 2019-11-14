import string
from nltk.translate import bleu_score
from nltk.tokenize import TweetTokenizer


def calc_bleu_many(cand_seq, ref_sequences):
    sf = bleu_score.SmoothingFunction()
    return bleu_score.sentence_bleu(ref_sequences, cand_seq,
                                    smoothing_function=sf.method1,
                                    weights=(0.5, 0.5))


def calc_bleu(cand_seq, ref_seq):
    return calc_bleu_many(cand_seq, [ref_seq])


def tokenize(s):
    return TweetTokenizer(preserve_case=False).tokenize(s)


def untokenize(words):
    to_pad = lambda t: not t.startswith("'") and \
                       t not in string.punctuation
    return "".join([
        (" " + i) if to_pad(i) else i
        for i in words
    ]).strip()
