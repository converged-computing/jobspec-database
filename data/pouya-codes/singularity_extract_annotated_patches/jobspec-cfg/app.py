import logging

from submodule_utils.logging import logger_factory
from extract_annotated_patches.parser import create_parser
from extract_annotated_patches import *

logger_factory()
logger = logging.getLogger('extract_annotated_patches')

if __name__ == "__main__":
    parser = create_parser()
    config = parser.get_args()
    ape = AnnotatedPatchesExtractor(config)
    ape.run()
