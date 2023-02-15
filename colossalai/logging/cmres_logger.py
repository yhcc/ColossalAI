import logging
from logging.handlers import RotatingFileHandler

from colossalai.logging.cmreslogging.handlers import CMRESHandler

# Default elastic search server host
DEFAULT_ES_HOST = "http://10.140.0.75:9200"
# Default log file name
DEFAULT_FILE_NAME = "python_cmres.log"


def setup_logger(
    logger_name: str,
    log_level: int = logging.INFO,
    file_name: str = DEFAULT_FILE_NAME,
) -> logging.Logger:
    """Configure the logger that is used for uniscale framework.

    Args:
        logger_name (str): Used to create or get the correspoding logger in
            getLogger call. It will get the root logger by default.
        log_level : Default is logging.DEBUG.
        file_name (str): Log file name, default is "python_es.log".

    Returns:
        logger (logging.Logger): the created or modified logger.

    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    es_handler = CMRESHandler(hosts=[DEFAULT_ES_HOST])
    es_handler.setLevel(logging.DEBUG)
    logger.addHandler(es_handler)

    file_handler = RotatingFileHandler(filename=file_name)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    return logger
