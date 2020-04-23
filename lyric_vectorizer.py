import os
import yaml
import pickle
import logging
from helpers import setup_logger
from generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment
import numpy as np

from vectorizer import Vectorizer
setup_logger()
logger = logging.getLogger(__name__)


class LyricVectorizer(Vectorizer):

    def __init__(self, gen_env=None):
        super().__init__(gen_env)

        self.name = __name__

    def vectorize_token_list(self, token_list):

        logger.info(
            'Vectorizing lists of tokens and adding to word_to_vec dictionary.')
        n_failed = 0
        for tokens in token_list:
            if not tokens:
                logger.warning(f'Empty tokens list. Skipping...')
                continue
            success_pct = self.memoize_vectorize_tokens(tokens)
            if success_pct != 100:
                n_failed += 1
        logger.info(f'All lines\' tokens have been vectorized')
        if n_failed > 0:
            logger.warning('Some words could not be converted.')
        return self.word_to_vec

    def vectorize_line(self, line):
        return [self.vectorize_word(w) for w in line]

    def vectorize_lines(self, token_list, start, stop):
        return [self.vectorize_line(l) for l in token_list[start:stop]]

    def vectorize_song(self, token_list):
        return self.vectorize_lines(token_list, 0, len(token_list))

    def load_song(self, songname):
        return self.load(songname)

    def save_song(self, songname):
        return self.save(songname)
