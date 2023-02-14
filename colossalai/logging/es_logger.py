import logging

from colossalai.logging.cmreslogging.handlers import CMRESHandler

# Default elastic search server host
DEFAULT_ES_HOST = "http://10.140.0.75:9200"


def setup_logger(logger_name: str) -> logging.Logger:
    """Configure the logger that is used for uniscale framework.

    Args:
        logger_name (str): Used to create or get the correspoding logger in
            getLogger call. It will get the root logger by default.

    Returns:
        logger (logging.Logger): the created or modified logger.

    """

    handler = CMRESHandler(hosts=[DEFAULT_ES_HOST])
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger
