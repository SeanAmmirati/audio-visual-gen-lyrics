
import logging
from helpers import setup_logger, dict_assign, find_first_file_with_ext

from generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment
from tokenizer import Tokenizer

import re
import os

import pylrc


from generatorio import PickleGeneratorIO, YAMLGeneratorIO
setup_logger()
logger = logging.getLogger(__name__)


class LyricTokenizer(Tokenizer):

    def __init__(self, gen_env=None):
        super().__init__(gen_env)

        self.lrc_str = None
        self.lrc_obj = None
        self.lyric_list = None
        self.tokens_list = None
        self.name = __name__

        self.attrs = ['tokens_list', 'lrc_str', 'lyric_list', 'lrc_obj']

    def tokenize_lyrics(self, songname, process=True):

        self.lrc_str = self.env.read_lrc_file(songname)
        self.lrc_obj = pylrc.parse(self.lrc_str)

        self.lyric_list = self.lrc_to_lyric_list(self.lrc_obj)
        logger.info(f'Tokenizing lyrics for song {songname}')

        tokens_list = [self.tokenize_phrase(lyric_line)
                       for lyric_line in self.lyric_list]
        logger.debug('Generated token list from lines of lrc file.')

        self.tokens_list = tokens_list
        return tokens_list

    def lrc_to_lyric_list(self, lrc_obj):
        """Returns a generator with each line of lyrics, with no timestamps.

        Arguments:
            lrc_file_path {str} -- A string representation of a valid lrc file format.
        """
        logger.debug('Converting lrc file to list of lyric strings.')

        return [x.text for x in lrc_obj]
