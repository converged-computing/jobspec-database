import logging

from submodule_utils.logging import logger_factory
from auto_annotate.parser import create_parser
from auto_annotate import *
from submodule_utils import set_random_seed

logger_factory()
logger = logging.getLogger('auto_annotate')

if __name__ == "__main__":
    parser = create_parser()
    config = parser.get_args()
    set_random_seed(config.seed)
    app = AutoAnnotator.from_log_file(config)
    app.run()
