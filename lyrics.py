import logging
from helpers import setup_logger

from lyric_tokenizer import LyricTokenizer
from lyric_vectorizer import LyricVectorizer

from lyric_line_assigner import LyricLineAssigner

from generator_object import GeneratorObject

from image_categories import ImageCategories

from pylrc import parser

import pandas as pd

setup_logger()
logger = logging.getLogger(__name__)


class Lyrics(GeneratorObject):

    def __init__(self, songname, tokenizer=None, vectorizer=None, gen_env=None):

        super().__init__(gen_env)
        self.name = __name__ if __name__ != '__main__' else 'lyrics'

        self.songname = songname

        if not tokenizer:
            logger.info('No tokenizer specified. Using default tokenizer.')
            self.tokenizer = LyricTokenizer(self.env)
        else:
            logger.info('Using custom tokenizer class.')
            if not isinstance(tokenizer, LyricTokenizer):
                logger.error(
                    'You must use an instance of the tokenizer class as a tokenizer. See lyric_tokenizer.py')
                raise ValueError('Invalid tokenizer')
            else:
                self.tokenizer = tokenizer

        if not vectorizer:
            logger.info('No vectorizer specified. Using default vectorizer.')
            self.vectorizer = LyricVectorizer(self.env)
        else:
            logger.info('Using custom vectorizer class.')
            if not isinstance(vectorizer, LyricVectorizer):
                logger.error(
                    'You must use an instance of the vectorizer class as a vectorizer. See lyric_vectorizer.py')
                raise ValueError('Invalid vectorizer')
            else:
                self.vectorizer = vectorizer

        self._tokens = None
        self._word_to_vec = None
        self._lyric_list = None
        self._lrc_str = None
        self._lrc_obj = None
        self.topics = None
        self.name = __name__ if __name__ != '__main__' else 'lyrics'
        self.attrs = ['_tokens', '_word_to_vec',
                      '_lyric_list', '_lrc_str', '_lrc_obj']

    def generate_tokens(self, load=True, save=True):

        try:
            assert load
            self.tokenizer.load(self.songname)
            logger.info(
                f'Could not find saved tokens list in {self.env.LYRIC_PATH}. Generating one now.')
        except (FileNotFoundError, AssertionError):
            self.tokenizer.tokenize_lyrics(self.songname)
            if save:
                self.tokenizer.save(self.songname)
                logger.info(
                    f'Saved tokens list to {self.env.LYRIC_PATH} directory.')
        logger.info('Generated tokens.')

    @property
    def lrc_obj(self):
        if not self._lrc_obj:

            self._lrc_obj = self.tokenizer.lrc_obj
        return self._lrc_obj

    @property
    def tokens(self):
        if self._tokens:
            return self._tokens
        else:
            self.generate_tokens()
            self._tokens = self.tokenizer.tokens_list
            return self._tokens

    @property
    def lyric_list(self):
        if self._lyric_list:
            return self._lyric_list
        else:
            self._lyric_list = self.tokenizer.lyric_list
            return self._lyric_list

    @property
    def lrc_str(self):
        if self._lrc_str:
            return self._lrc_str
        else:
            self._lrc_str = self.tokenizer.lrc_str
            return self._lrc_str

    @property
    def word_to_vec(self):
        if self._word_to_vec:
            return self._word_to_vec
        else:
            self.generate_wordvecs()
            self._word_to_vec = self.vectorizer.word_to_vec
            return self._word_to_vec

    def generate_wordvecs(self, save=True):
        if not self.tokens:
            self.generate_tokens(load=False)

        try:
            self.vectorizer.load(self.songname)
        except FileNotFoundError:
            logger.info(
                f'Could not find saved word_to_vec dictionary in {self.env.SONG_EMBEDDING_PATH}. Generating one now.')
            self.vectorizer.vectorize_token_list(self.tokens)
            if save:
                self.vectorizer.save(self.songname)
                logger.info(
                    f'Saved vectorized dictionary to {self.env.SONG_EMBEDDING_PATH}.')
        else:
            logger.info(
                f'Loaded word_to_vec dictionary from {self.env.SONG_EMBEDDING_PATH}.')
        logger.info('Vectorized lyrics.')

    def vectorized_song(self):
        if not self._word_to_vec:
            self.generate_wordvecs()

        return self.vectorizer.vectorize_song(self.tokens)

    def sort_topics(self, image_categories=None, *args, **kwargs):
        if not image_categories:
            image_categories = ImageCategories(gen_env=self.env)
        line_assigner = LyricLineAssigner(*args, **kwargs)
        vectorized_song = self.vectorized_song()

        for i, line in enumerate(vectorized_song):
            if not line:
                self.lrc_obj[i].category_id = None
                continue
            self.lrc_obj[i].category_id = line_assigner.assign_line(
                line, image_categories.vectors, None)
        return self.lrc_obj

    def assign_topics(self, image_categories=None, n=1, *args, **kwargs):
        if not self.lrc_obj:
            self.sort_topics(image_categories, *args, **kwargs)
        if not all(hasattr(x, 'category_id') for x in self.lrc_obj):
            self.sort_topics(image_categories, *args, **kwargs)
        for lrc_obj in self.lrc_obj:
            if not lrc_obj.category_id:
                lrc_obj.topic_ids = None
                continue
            lrc_obj.topic_ids = lrc_obj.category_id[0:n]

    def generate_lyric_df(self):
        temp_list = []
        for lrc_o in self.lrc_obj:
            data = dict(time=pd.to_timedelta(lrc_o.time, unit='s'),
                        lyrics=lrc_o.text)
            if lrc_o.topic_ids:
                for i, topic in enumerate(lrc_o.topic_ids):
                    data['topic'] = i
                    data['topic_id'] = topic

                    temp_list.append(data.copy())
        df = pd.DataFrame(temp_list).dropna(how='any')
        df['topic_id'] = df['topic_id'].astype(int)
        return df

    def save(self):
        self.genio.save(self.songname)

    def load(self):
        self.genio.load(self.songname)


if __name__ == '__main__':

    image_categories = ImageCategories()
    lyrics = Lyrics('a_loving_feeling')

    lyrics.load()
    lyrics.assign_topics(n=5)

    print([l.topic_ids for l in lyrics.lrc_obj])
    df = lyrics.generate_lyric_df()
    import pdb
    pdb.set_trace()
