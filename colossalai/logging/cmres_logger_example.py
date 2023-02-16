import time

from colossalai.logging.cmres_logger import setup_logger

logger = setup_logger(__file__)

if __name__ == "__main__":
    logger.info("info for cmres-logging")
    logger.warning("warning for cmres-logging")
    logger.debug("debug for cmres-logging")
    logger.error("error for cmres-logging")

    time.sleep(1)
