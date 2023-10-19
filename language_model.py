import math
import pynlpl.lm.lm
from nltk import ngrams


class LanguageModel:
    def __init__(self, model_path):
        self.model = pynlpl.lm.lm.ARPALanguageModel(filename=model_path, mode='simple')

    def get_n_grams(self, sentence, n):
        ss = sentence.split()
        n_grams = []
        if len(ss) < n:
            n_grams = self.get_n_grams(sentence, n - 1)
        else:
            for n_gram in ngrams(ss, n, pad_left=True, pad_right=True, left_pad_symbol=None, right_pad_symbol=None):
                n_grams.append(tuple([i for i in n_gram if i is not None]))
        return n_grams

    def get_sentence_score(self, sentence, n=3, unknown_word_prob=1e-5, normalization=True):
        sentence = sentence.lower()
        score = 0.
        n_grams = self.get_n_grams(sentence, n)
        if len(n_grams) > 0:
            for n_gram in n_grams:
                try:
                    log_prob = self.model.ngrams.prob(n_gram)
                    score += math.exp(log_prob)
                except KeyError:
                    score += unknown_word_prob
            if normalization:
                return score / len(n_grams)
            else:
                return score
        else:
            raise ValueError('Empty sentence')
