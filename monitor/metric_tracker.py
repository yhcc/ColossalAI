import os
import socket
import time
from threading import Thread

import GPUtil
import psutil

from monitor import elastic_search as es


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
            job_id = "none"
            if os.getenv("SLURM_JOB_ID") is not None:
                job_id = os.getenv("SLURM_JOB_ID")

            job_name = "none"
            if os.getenv("SLURM_JOB_NAME") is not None:
                job_name = os.getenv("SLURM_JOB_NAME")

            key = f"{job_id}_{job_name}"

            hostname = socket.gethostname()
            timestamp = int(time.time())

            cpu_util = psutil.cpu_percent()

            mem = psutil.virtual_memory()
            mem_util = mem[2]

            gpu_rank = "none"
            if os.getenv("SLURM_PROCID") is not None:
                gpu_rank = int(os.getenv("SLURM_PROCID"))

            local_gpu_rank = "none"
            if os.getenv("SLURM_LOCALID") is not None:
                local_gpu_rank = os.getenv("SLURM_LOCALID")

            gpu_util = "none"
            gpu_mem_util = "none"
            if local_gpu_rank != "none":
                local_gpu_rank = int(local_gpu_rank)

                gpus = GPUtil.getGPUs()
                # TODO: need to check the whether the rank is related to the order of
                # the gpu list
                gpu = gpus[local_gpu_rank]
                gpu_util = gpu.load * 100
                gpu_mem_util = gpu.memoryUtil * 100

            metric_dict = {
                "key": key,
                "timestamp": timestamp,
                "hostname": hostname,
                "cpu_util": cpu_util,
                "mem_util": mem_util,
                "gpu_rank": gpu_rank,
                "local_gpu_rank": local_gpu_rank,
                "gpu_util": gpu_util,
                "gpu_mem_util": gpu_mem_util,
            }

            es.put_metric_to_elastic_search(metric_dict)

            time.sleep(self.interval)

    def stop(self):
        """
        Stop the metric tracker.
        """

        self.stopped = True
