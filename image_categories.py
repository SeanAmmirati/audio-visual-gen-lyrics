import logging
from helpers import setup_logger

from image_category_tokenizer import ImageCategoryTokenizer
from image_category_vectorizer import ImageCategoryVectorizer

from generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment

setup_logger()
logger = logging.getLogger(__name__)


class ImageCategories:

    def __init__(self, tokenizer=None, vectorizer=None, gen_env=None):
        logger.debug('Started initialization of Lyric class.')

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
        self.img_cat_tokenizer = ImageCategoryTokenizer(
            self.env) if not tokenizer else tokenizer
        self.img_cat_vectorizer = ImageCategoryVectorizer(
            self.env) if not vectorizer else vectorizer

        self._tokens = None
        self._vectors = None
        self._strings = None

    @property
    def strings(self):
        if not self._strings:
            self.img_cat_tokenizer.load_image_classes()
            self._strings = self.img_cat_tokenizer.image_classes
        return self._strings

    @property
    def tokens(self, save=True):
        if not self._tokens:
            try:
                self.img_cat_tokenizer.load()
            except FileNotFoundError:
                self.img_cat_tokenizer.tokenize_image_classes()
                if save:
                    self.img_cat_tokenizer.save()
            self._tokens = self.img_cat_tokenizer.class_tokens
        return self._tokens

    @property
    def vectors(self, save=True):
        if not self._vectors:
            try:
                self.img_cat_vectorizer.load()
            except FileNotFoundError:
                self.img_cat_vectorizer.vectorize_categories(self.tokens)
                if save:
                    self.img_cat_vectorizer.save()
            self._vectors = self.img_cat_vectorizer.vectorized_dict

        return self._vectors

    def find_category_string_by_id(self, id_):
        return self.strings[id_]

    def find_category_tokens_by_id(self, id_):
        return self.tokens[id_]

    def find_category_vector_by_id(self, id_):
        return self.vectors[id_]


if __name__ == '__main__':
    img_cats = ImageCategories()

    string = img_cats.find_category_string_by_id(3)
    tokens = img_cats.find_category_tokens_by_id(3)
    vec = img_cats.find_category_vector_by_id(3)

    print(string, tokens, vec)
