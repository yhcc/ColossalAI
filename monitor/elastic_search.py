from elasticsearch import Elasticsearch

from colossalai.logging import get_dist_logger
from monitor.types import ES_METRIC_SUBMIT_URL, MetricType

es = Elasticsearch(hosts=[ES_METRIC_SUBMIT_URL], request_timeout=30)
logger = get_dist_logger(name="colossalai_monitor")


def put_metric_to_elastic_search(metric_type: MetricType, metric_value: dict) -> bool:
    """
    Creates or updates a document in an index.

    Args:
        metric_type (MetricType): Metric type of the document.
        metric_value (dict): The dict format metric value of the document.

    Returns:
        (bool) is the status of put metric to elastic search.

    """

    try:
        logger.debug(f"Submit metrics to ES: {metric_type.value}: {metric_value}")
        response = es.index(index=metric_type.value, document=metric_value, timeout="30s")
        logger.debug(message=response)
        return True
    except Exception as error:    # pylint: disable=broad-except
        logger.error(message=error)
        return False
