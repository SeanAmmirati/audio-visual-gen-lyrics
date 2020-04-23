
import logging
from helpers import setup_logger

from tokenizer import Tokenizer
from image_category_vectorizer import ImageCategoryVectorizer


setup_logger()
logger = logging.getLogger(__name__)


class ImageCategoryTokenizer(Tokenizer):
    def __init__(self, gen_env=None):
        super().__init__(gen_env)

        self.name = __name__ if __name__ != '__main__' else 'image_category_tokenizer'
        self.image_classes = None
        self.class_tokens = None

        self.attrs = ['class_tokens']

    def tokenize_category(self, category, cat_sep=','):
        seperated = category.split(',')
        return [self.tokenize_phrase(phrase) for phrase in seperated]

    def load_image_classes(self):
        self.image_classes = self.env.read_id_to_img_class()

    def tokenize_image_classes(self):

        if not self.image_classes:
            logger.debug('No image classes loaded. Attempting to load now.')
            self.load_image_classes()

        self.class_tokens = {id_: self.tokenize_category(
            cat) for id_, cat in self.image_classes.items()}


if __name__ == '__main__':
    im_token = ImageCategoryTokenizer()
    # im_token.tokenize_image_classes()
    # im_token.save()
    im_token.load()

    im_vectorizer = ImageCategoryVectorizer()
    print(im_vectorizer.vectorize_categories(im_token.class_tokens))
    im_vectorizer.save()
