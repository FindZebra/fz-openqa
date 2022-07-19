import os
import subprocess
import time
from copy import copy

from loguru import logger

from es_status import ping_es


class ElasticSearchInstance(object):
    TIMEOUT = 3600

    def __init__(self, disable: bool = False, **kwargs):
        self.disable = disable
        self.kwargs = copy(kwargs)

    def __enter__(self):
        # make a database connection and return it
        if ping_es():
            logger.info("Elasticsearch is already running")
            return

        if not self.disable:
            env = copy(os.environ)
            cmd = "elasticsearch"
            logger.info(
                f"Spawning ElasticSearch: {cmd}, "
                f"ES_JAVA_OPTS={env.get('ES_JAVA_OPTS', '<none>')}"
            )
            self.es_proc = subprocess.Popen([cmd], env=env, **self.kwargs)
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
