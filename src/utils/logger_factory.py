import os
import logging
from logging.config import dictConfig
import yaml

import utils


class LoggerFactory:

    @staticmethod
    def get_logger(module_name=None):
        # Get Configuration file path
        os.makedirs(name=utils.ApplicationPaths.logs(), exist_ok=True)
        logging.config.dictConfig(yaml.load(open(utils.ApplicationPaths.config("logging.yaml"), 'r')))

        if module_name:
            return logging.getLogger(module_name)
        return logging.getLogger()
