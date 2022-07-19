import subprocess
import time
from copy import copy
from typing import Optional

from loguru import logger

from es_status import ping_es


class ElasticSearchInstance(object):
    TIMEOUT = 3600

    def __init__(self, disable: bool = False, es_args: Optional[str] = None, **kwargs):
        self.disable = disable
        self.es_args = es_args
        self.kwargs = copy(kwargs)

    def __enter__(self):
        # make a database connection and return it
        if ping_es():
            logger.info("Elasticsearch is already running")
            return

        if not self.disable:
            cmd = "elasticsearch"
            if self.es_args is not None:
                cmd = f"{self.es_args} {cmd}"
            logger.info(f"Spawning ElasticSearch: {cmd}")
            self.es_proc = subprocess.Popen([cmd], **self.kwargs)
            t0 = time.time()
            while not ping_es():
                time.sleep(0.5)
                if time.time() - t0 > self.TIMEOUT:
                    raise TimeoutError("Couldn't ping the ES instance.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if hasattr(self, "es_proc"):
            logger.info("Terminating elasticsearch process")
            self.es_proc.terminate()
