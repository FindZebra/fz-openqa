import os
from pathlib import Path

from fz_openqa.utils.datastruct import PathLike


def display_file_size(key: str, fn: PathLike, print_fn=None):
    if print_fn is None:
        print_fn = print
    fn = Path(fn)
    if fn.is_dir():
        print(f"{key}:")
        for f in fn.iterdir():
            display_file_size(f"{f} -- {os.path.basename(fn.name)}", f, print_fn)
    else:
        s = os.path.getsize(fn)
        s /= 1024 ** 3
        msg = f"{key} - disk_size={s:.3f} GB"
        print_fn(msg)
