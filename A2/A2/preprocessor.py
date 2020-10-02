import nltk
from functools import lru_cache

# nltk.download('punkt')


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=100000)(nltk.PorterStemmer().stem)
        self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize
        # self.tokenize = nltk.tokenize.word_tokenize
        # self.tokenize = nltk.tokenize.SpaceTokenizer().tokenize

    def __call__(self, text):
        tokens = self.tokenize(text)
        tokens = [self.stem(token) for token in tokens]
        return tokens
