from enum import Enum

# Elasticsearch metric submit URL
ES_METRIC_SUBMIT_URL = "http://10.140.0.75:9200"


# Metric types
class MetricType(Enum):
    """
    Resource metric types.
    """

    NODE_CPU_UTIL = "NODE_CPU_UTIL"
    NODE_MEM_TOTAL = "NODE_MEM_TOTAL"
    NODE_MEM_AVAIL = "NODE_MEM_AVAIL"
    NODE_MEM_UTIL = "NODE_MEM_UTIL"
    GPU_ID = "GPU_ID"
    GPU_UTIL = "GPU_UTIL"
    GPU_MEM_UTIL = "GPU_MEM_UTIL"
