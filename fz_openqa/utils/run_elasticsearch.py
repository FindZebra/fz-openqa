from os import popen
from pathlib import Path
import subprocess
import time

print(">> Starting Elastic Search...")
subprocess.Popen([str(Path.home())+'/elasticsearch-7.14.1/bin/elasticsearch','-d'])
time.sleep(5)
print(">> Elastic Search is running!")