import os
import subprocess
import time
from os import popen
from pathlib import Path

all_subdirs = [d for d in os.listdir(str(Path.home())) if "elasticsearch" in d]
version = all_subdirs[0].split("-")[-1]

print(">> Starting Elastic Search...")
subprocess.Popen(
    [
        str(Path.home()) + "/elasticsearch-" + version + "/bin/elasticsearch",
        "-d",
    ]
)
time.sleep(5)
print(">> Elastic Search is running!")
