from generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment
from generatorio import PickleGeneratorIO, YAMLGeneratorIO
from helpers import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)


class GeneratorObject:

    def __init__(self, gen_env=None):
        logger.debug(
            f'Started initialization of {self.__class__.__name__}class.')

        if not gen_env:
            logger.info(
                'No passed environment class. Defaulting to Wikipedia2Vec and BigGAN.')

            gen_env = WikipediaBigGANGenerationEnviornment()

        elif __name__ != '__main__':
            logger.debug('Passed environment class from other class')

        else:
            if not isinstance(gen_env, GenerationEnvironment):
                logger.error(
                    'Argument gen_env is not a GenerationEnvironment instance. You must pass the appropriate object here.')
                raise ValueError('Not a Generation Environment instance.')
            else:
                logger.info('Custom enviornment class passed.')

        self.env = gen_env

        if self.env.SAVE_FILETYPE == 'pickle':
            self.genio = PickleGeneratorIO(
                self, self.env)
        elif self.env.SAVE_FILETYPE == 'yaml':
            self.genio = YAMLGeneratorIO(self, self.env)
