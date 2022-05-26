import sys

from elasticsearch import Elasticsearch

from loguru import logger
import time
import socket


def ping_es(host=None, **kwargs):
    if host is None:
        hosts = None
    else:
        hosts = [host]
    return Elasticsearch(hosts=hosts).ping(**kwargs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        host = "localhost"
    else:
        host = str(sys.argv[1])
    logger.info(f"Pinging {host}")
    while True:
        logger.info(f"({socket.gethostname()} -> {host}) ping: {ping_es(host, pretty=True, human=True)}")
        time.sleep(1)
