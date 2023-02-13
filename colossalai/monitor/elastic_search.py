from elasticsearch import Elasticsearch

from colossalai.logging import get_dist_logger
from monitor.types import ES_INDEX

logger = get_dist_logger(name=ES_INDEX)


def put_metric_to_elastic_search(elastic_search: Elasticsearch, metric_value: dict) -> bool:
    """
    Creates or updates a document in an index.

    Args:
        elastic_search (Elasticsearch): The initialized elastic search HTTP request.
        metric_value (dict): The dict format metric value of the document.

    Returns:
        (bool) is the status of put metric to elastic search.

    """

    try:
        logger.debug(f"Submit metrics to ES index [{ES_INDEX}]: {metric_value}")
        response = elastic_search.index(index=ES_INDEX, document=metric_value, timeout="30s")
        logger.debug(message=response)
        return True
    except Exception as error:  # pylint: disable=broad-except
        logger.error(message=error)
        return False
