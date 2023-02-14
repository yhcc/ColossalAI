import time

from colossalai.logging.es_logger import setup_logger

logger = setup_logger(__file__)

if __name__ == "__main__":
    logger.info("info for es-logging")
    logger.debug("debug for es-logging")
    logger.error("error for es-logging")
    logger.info("test for es-logging")

    time.sleep(1)
