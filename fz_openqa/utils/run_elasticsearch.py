import subprocess
import time
from os import popen
from pathlib import Path

print(">> Starting Elastic Search...")
subprocess.Popen(
    [str(Path.home()) + "/elasticsearch-7.14.1/bin/elasticsearch", "-d"]
)
time.sleep(5)
print(">> Elastic Search is running!")
