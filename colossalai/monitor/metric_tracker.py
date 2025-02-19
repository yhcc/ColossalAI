import os
import re
import socket
import time
from datetime import datetime
from threading import Thread

import GPUtil
import psutil
import pytz
from elasticsearch import Elasticsearch

from colossalai.monitor import elastic_search as es_util


class MetricTracker(Thread):
    """
    Track resource usage during task training.

    Args:
        es_server_address (str): The server address of elastic search service.
        interval (float): The tracking interval. By default, 15 seconds.

    """

    def __init__(self, es_server_address: str, interval: float = 15):
        super(MetricTracker, self).__init__()
        self.elastic_search = Elasticsearch(hosts=[es_server_address], request_timeout=30)
        self.stopped = False
        self.interval = interval
        self.start()

    def run(self):
        """
        Run the metric tracker.
        """

        def format_2_decimal(num: float) -> float:
            return float(f"{num:.2f}")

        while not self.stopped:
            try:
                job_id = "none"
                if os.getenv("SLURM_JOB_ID") is not None:
                    job_id = os.getenv("SLURM_JOB_ID")

                job_name = "none"
                if os.getenv("SLURM_JOB_NAME") is not None:
                    job_name = os.getenv("SLURM_JOB_NAME")

                key = f"{job_id}_{job_name}"

                hostname = socket.gethostname()

                time_zone = pytz.timezone("Asia/Shanghai")
                timestamp = datetime.now(time_zone)

                cpu_util = format_2_decimal(psutil.cpu_percent())

                mem = psutil.virtual_memory()
                mem_util = format_2_decimal(mem[2])

                gpu_rank = "none"
                if os.getenv("SLURM_PROCID") is not None:
                    gpu_rank = int(os.getenv("SLURM_PROCID"))

                local_gpu_rank = "none"
                if os.getenv("SLURM_LOCALID") is not None:
                    local_gpu_rank = int(os.getenv("SLURM_LOCALID"))

                flops = 0.0
                if os.getenv("FLOPS") is not None:
                    flops = float(os.getenv("FLOPS"))

                metric_dict = {
                    "key": key,
                    "timestamp": timestamp,
                    "hostname": hostname,
                    "cpu_util": cpu_util,
                    "mem_util": mem_util,
                    "gpu_rank": gpu_rank,
                    "local_gpu_rank": local_gpu_rank,
                    "flops": flops,
                    "gpu_info": "none",
                }

                raw_network_io = psutil.net_io_counters(pernic=True)
                for interface in raw_network_io:
                    if re.search("^ib[0-9]+", interface):
                        raw_network_io_info = raw_network_io[interface]
                        metric_dict[interface] = {
                            "bytes_sent": raw_network_io_info.bytes_sent,
                            "bytes_received": raw_network_io_info.bytes_recv,
                            "packets_sent": raw_network_io_info.packets_sent,
                            "packets_received": raw_network_io_info.packets_recv,
                            "error_in": raw_network_io_info.errin,
                            "error_out": raw_network_io_info.errout,
                            "drop_in": raw_network_io_info.dropin,
                            "drop_out": raw_network_io_info.dropout,
                        }

                if os.getenv("CUDA_VISIBLE_DEVICES") is not None and isinstance(local_gpu_rank, int):
                    # Get the GPU device list in string format
                    gpu_device_ids = os.getenv("CUDA_VISIBLE_DEVICES")
                    gpu_device_id_list = gpu_device_ids.split(",")

                    # Get the GPU device ID based on the local rank ID
                    device_id = int(gpu_device_id_list[local_gpu_rank])

                    # Get the GPU info in this node
                    gpus = GPUtil.getGPUs()
                    selected_gpu = gpus[device_id]

                    gpu_device_id = selected_gpu.id
                    gpu_name = selected_gpu.name
                    gpu_util = format_2_decimal(selected_gpu.load * 100)
                    gpu_mem_util = format_2_decimal(selected_gpu.memoryUtil * 100)

                    metric_dict["gpu_info"] = {
                        "gpu_device_id": gpu_device_id,
                        "gpu_name": gpu_name,
                        "gpu_util": gpu_util,
                        "gpu_mem_util": gpu_mem_util,
                    }

                es_util.put_metric_to_elastic_search(self.elastic_search, metric_dict)
            except ValueError:
                continue

            time.sleep(self.interval)

    def stop(self):
        """
        Stop the metric tracker.
        """

        self.stopped = True
