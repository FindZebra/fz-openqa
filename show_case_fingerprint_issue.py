import hashlib
import json
import os
from typing import Dict, Any
from rich import print
import numpy as np
from datasets import load_dataset, Dataset
Dataset.map

Batch = Dict[str, Any]
filename = 'example.json'


class Transformation():
    """A transformation with a random state that cannot be fingerprinted"""

    def __init__(self):
        self.state = np.random.random()

    def __call__(self, batch: Batch) -> Batch:
        batch['x'] = [np.random.random() for _ in batch['x']]
        return batch


def generate_dataset():
    """generate a simple dataset"""
    rgn = np.random.RandomState(24)
    data = {
        'data': [{'x': float(y), 'y': -float(y)} for y in
                 rgn.random(size=(1000,))]}
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(data))

    return filename


def process_dataset_with_cache(num_proc=1, remove_cache=False,
                               cache_expected_to_exist=False):

    # load the generated dataset
    dset: Dataset = next(
        iter(load_dataset('json', data_files=filename, field='data').values()))
    new_fingerprint = hashlib.md5("static-id".encode("utf8")).hexdigest()

    # get the expected cached path
    cache_path = dset._get_cache_file_path(new_fingerprint)
    if remove_cache and os.path.exists(cache_path):
        os.remove(cache_path)

     # check that the cache exists, and print a statement
    # if was actually expected to exist
    cache_exist = os.path.exists(cache_path)
    print(f"> cache file exists={cache_exist}")
    if cache_expected_to_exist and not cache_exist:
        print("=== Cache does not exist! ====")

    # apply the transformation with the new fingerprint
    dset = dset.map(
        Transformation(),
        batched=True,
        num_proc=num_proc,
        new_fingerprint=new_fingerprint,
        desc="mapping dataset with transformation")


generate_dataset()

for num_proc in [1, 2]:
    print(f"# num_proc={num_proc}, first pass")
    # first pass to generate the cache (always create a new cache here)
    process_dataset_with_cache(remove_cache=True,
                               num_proc=num_proc,
                               cache_expected_to_exist=False)
    print(f"# num_proc={num_proc}, second pass")
    # second pass, expects the cache to exist
    process_dataset_with_cache(remove_cache=False,
                               num_proc=num_proc,
                               cache_expected_to_exist=True)

os.remove(filename)
