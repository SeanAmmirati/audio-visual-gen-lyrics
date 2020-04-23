import os
import yaml
import pickle
import logging
from helpers import setup_logger
from generator_object import GeneratorObject
from generatorio import PickleGeneratorIO, YAMLGeneratorIO

import numpy as np
setup_logger()
logger = logging.getLogger(__name__)


class Vectorizer(GeneratorObject):

    def __init__(self, gen_env=None):
        super().__init__(gen_env)
        self.word_embedder = self.env.word_embedder()

        self.word_to_vec = {}
        self.name = __name__

        self.attrs = ['word_to_vec']

        if self.env.SAVE_FILETYPE == 'pickle':
            self.genio = PickleGeneratorIO(
                self, self.env)
        elif self.env.SAVE_FILETYPE == 'yaml':
            self.genio = YAMLGeneratorIO(self, self.env)

    def memoize_vectorize_tokens(self, tokens):

        logger.debug(
            f'Tokens: {tokens} -- Vecorizing tokens in token list.')

        success = 0
        n = len(tokens)
        for i, t in enumerate(tokens):
            if t in self.word_to_vec:
                logger.debug(f'Token: {t} -- Already converted. Skipping...')
                success += 1
                continue

            logger.debug(
                f'Token: {t} -- Converting to word vector ({i + 1} out of {len(tokens)}.')
            try:
                self.word_to_vec[t] = self.word_embedder.get_word_vector(t)
            except Exception as e:
                logger.warning(f'Token: {t} -- Could not find in vocabulary.')
            else:
                logger.debug(f'Token: {t} - - Successfully vectorized token.)')
                success += 1

        success_pct = round((success / n) * 100, 2)
        if success_pct != 100:
            logger.warning(
                f'Tokens: {tokens} -- Only {success_pct}% of tokens successfully converted.')
        else:
            logger.debug(
                f'Tokens: {tokens} -- All tokens were converted.')
        logger.debug(f'Tokens: {tokens} -- Vectorization complete.')

        return success_pct

    def vectorize_word(self, word):

        try:
            ret = self.word_to_vec[word]
        except KeyError:
            logger.debug(
                f'Did not find word \'{word}\' in generated dictionary. Using embedder.')
            ret = self.word_embedder.get_word_vector(word)
        else:
            logger.debug(f'Found word \'{word}\' in generated dictionary.')
        return ret

    def save(self, dirname=None):
        """Exports generated vectorization dictionary for future use

        Keyword Arguments:
            export_type {'pickle' | 'yaml'}  Type to export to. Pickle may have more overhead in space, but yaml will have higher processing times on load.
            (default: {'pickle'})
        """
        self.genio.save(dirname)
        logging.info(
            f'Saved word vectorizations for {dirname}')

    def load(self, dirname=None):
        """Loads word to vector dictionary. If pickle exists, it will use that since it's easiest to load in.
        """
        self.genio.load(dirname)
        logging.info(f'Loaded word vectorizations at {dirname}')
