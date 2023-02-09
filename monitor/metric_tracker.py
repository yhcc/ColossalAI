import time
from threading import Thread

import GPUtil
import psutil

from monitor import elastic_search as es
from monitor.types import MetricType


class MetricTracker(Thread):
    """
    Track resource usage during task training.
    """

    def __init__(self, interval):
        super(MetricTracker, self).__init__()
        self.stopped = False
        self.interval = interval
        self.start()

    def run(self):
        """
        Run the metric tracker.
        """

        while not self.stopped:
            cpu_util = psutil.cpu_percent()
            cpu_util_dict = {"cpu_util": f"{cpu_util}%"}
            es.put_metric_to_elastic_search(MetricType.NODE_CPU_UTIL, cpu_util_dict)

            mem = psutil.virtual_memory()
            mem_total = mem[0]
            mem_available = mem[1]
            mem_util = mem[2]

            mem_total_dict = {"mem_total": f"{mem_total}Bytes"}
            mem_available_dict = {"mem_available": f"{mem_available}Bytes"}
            mem_util_dict = {"mem_util": f"{mem_util}%"}

            es.put_metric_to_elastic_search(MetricType.NODE_MEM_TOTAL, mem_total_dict)
            es.put_metric_to_elastic_search(MetricType.NODE_MEM_AVAIL, mem_available_dict)
            es.put_metric_to_elastic_search(MetricType.NODE_MEM_UTIL, mem_util_dict)

            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_id_dict = {"gpu_id": f"{gpu.id}"}
                gpu_load_dict = {"gpu_load": f"{gpu.load * 100}%"}
                gpu_mem_util_dict = {"gpu_mem_util": f"{gpu.memoryUtil * 100}%"}

                es.put_metric_to_elastic_search(MetricType.GPU_ID, gpu_id_dict)
                es.put_metric_to_elastic_search(MetricType.GPU_UTIL, gpu_load_dict)
                es.put_metric_to_elastic_search(MetricType.GPU_MEM_UTIL, gpu_mem_util_dict)

            time.sleep(self.interval)

    def stop(self):
        """
        Stop the metric tracker.
        """

        self.stopped = True
