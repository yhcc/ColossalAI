from enum import Enum


# metric type
class MetricType(Enum):
    NODE_CPU_UTIL = "NODE_CPU_UTIL"
    NODE_MEM_TOTAL = "NODE_MEM_TOTAL"
    NODE_MEM_AVAIL = "NODE_MEM_AVAIL"
    NODE_MEM_UTIL = "NODE_MEM_UTIL"
    GPU_UTIL = "GPU_UTIL"
    GPU_MEM_UTIL = "GPU_MEM_UTIL"
