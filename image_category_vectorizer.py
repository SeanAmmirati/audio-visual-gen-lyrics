import os
import yaml
import pickle
import logging
from helpers import setup_logger
from generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment
from generatorio import PickleGeneratorIO, YAMLGeneratorIO

from vectorizer import Vectorizer
import numpy as np
setup_logger()
logger = logging.getLogger(__name__)


class ImageCategoryVectorizer(Vectorizer):
    def __init__(self, gen_env=None):
        super().__init__(gen_env)

        self.name = __name__ if __name__ != '__main__' else 'image_category_vectorizer'
        self.vectorized_dict = None

        self.attrs = ['vectorized_dict']

    def _mean_strategy(self, category_tokens):

        wordvec_sum = np.zeros(self.env.wordvec_dim)
        n_phrases = 0

        for tokens in category_tokens:
            n = len(tokens)
            if n == 0:
                continue

            vec = np.zeros(self.env.wordvec_dim)
            n_vectorizable_phrases = 0
            for token in tokens:
                try:
                    vectorized = self.vectorize_word(token)
                except KeyError:
                    pass
                else:
                    n_vectorizable_phrases += 1
                    vec += vectorized
            if n_vectorizable_phrases == 0:
                continue
            else:
                n_phrases += 1
            vec = vec / n_vectorizable_phrases
            wordvec_sum += vec
        mean_wordvec = (
            wordvec_sum / n_phrases) if n_phrases != 0 else wordvec_sum

        return mean_wordvec

    def vectorize_category(self, category_tokens, strategy='mean'):
        if strategy == 'mean':
            return self._mean_strategy(category_tokens)

    def vectorize_categories(self, categories_tokens, strategy='mean'):
        self.vectorized_dict = {id_: self.vectorize_category(
            category) for id_, category in categories_tokens.items()}
        return self.vectorized_dict


if __name__ == '__main__':
    im_vec = ImageCategoryVectorizer()
    im_vec.load()
    print(im_vec.vectorized_dict)
