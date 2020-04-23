from abc import ABC, abstractmethod
from generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment
import logging
from helpers import setup_logger

import os

import pickle
import yaml
import numpy as np
setup_logger()
logger = logging.getLogger(__name__)


class GeneratorIO(ABC):

    def __init__(self, obj, gen_env=None):
        logger.debug('Started initialization of LyricVectorizer class.')

        if not gen_env:
            logger.info(
                'No passed enviornment class. Defaulting to Wikipedia2Vec and BigGAN.')

            gen_env = WikipediaBigGANGenerationEnviornment()

        else:
            if not isinstance(gen_env, GenerationEnvironment):
                logger.error(
                    'Argument gen_env is not a GenerationEnvironment instance. You must pass the appropriate object here.')
                raise ValueError('Not a Generation Environment instance.')
            else:
                logger.info('Custom enviornment class passed.')

        self.env = gen_env
        self.word_embedder = self.env.word_embedder()

        self.word_to_vec = {}
        self.obj = obj

    @property
    def attrs(self):
        return self.obj.attrs

    def get_saving_location(self, songname):

        if self.obj.name == 'lyric_tokenizer':
            self.save_loc = self.env.song_lyric_filename(songname)

        if self.obj.name == 'lyric_vectorizer':
            self.save_loc = self.env.song_embeddings_filename(songname)

        if self.obj.name == 'image_category_vectorizer':
            self.save_loc = self.env.class_embeddings_filename()

        if self.obj.name == 'image_category_tokenizer':
            self.save_loc = self.env.class_token_filename()

        if self.obj.name == 'lyrics':
            self.save_loc = self.env.complete_lyrics_filename(songname)

        return self.save_loc

    def make_save_locations(self, songname):

        full_path = self.get_saving_location(
            songname)
        dir_name = os.path.dirname(full_path)
        dir_hierarchy = []
        while dir_name:
            dir_hierarchy.append(dir_name)
            dir_name = os.path.dirname(dir_name)

        reversed_hierarchy = reversed(dir_hierarchy)

        for p in reversed_hierarchy:
            if not os.path.exists(p):
                logging.debug(f'No directory at {p}. Creating new directory.')
                os.mkdir(p)

    def transform_save(self, attr):
        return attr

    def transform_load(self, attr):
        return attr

    def save(self, songname=None):

        attrs = [getattr(self.obj, attr, None) for attr in self.attrs]

        transformed_attrs = [self.transform_save(x) for x in attrs]
        self.make_save_locations(songname)
        logger.debug(f'Saving {transformed_attrs} to {self.save_loc}')
        self.save_to_file(transformed_attrs, self.save_loc)
        logger.debug(
            f'Saved information for {self.obj.name} to {self.save_loc}')

    def load(self, songname=None):

        self.save_loc = self.get_saving_location(songname)
        try:
            res = self.load_from_file(self.save_loc)
        except FileNotFoundError:
            logger.error(f'No file to load in {self.save_loc}')
            raise
        for i, r in enumerate(res):
            logger.debug(f'Loading {r} into {self.attrs[i]} attribute.')
            setattr(self.obj, self.transform_load(self.attrs[i]), r)

        logger.info(
            f'Loaded information for {self.obj.name} from {self.save_loc}')

    @abstractmethod
    def load_from_file(self, where):
        pass

    @abstractmethod
    def save_to_file(self, x, where):
        pass


class PickleGeneratorIO(GeneratorIO):

    def save_to_file(self, x, where):
        with open(where, 'wb') as f:
            pickle.dump(x, f)

    def load_from_file(self, where):
        with open(where, 'rb') as f:
            return pickle.load(f)


class YAMLGeneratorIO(GeneratorIO):

    def save_to_file(self, x, where):
        with open(where, 'w') as f:
            yaml.dump(x, f)

    def load_from_file(self, where):
        with open(where, 'r') as f:
            return yaml.load(f)

    def transform_save(self, attr):
        if isinstance(attr, np.array):
            return attr.tolist()
        else:
            return attr

    def transform_load(self, attr):
        if isinstance(attr, list):
            return np.array(attr)
        else:
            return attr
