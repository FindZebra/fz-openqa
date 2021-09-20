# find . -type d -path "./*" -mmin +120 -exec rm -rf {} \;
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import rich


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--directory', default="/scratch/valv/raytune/",
                        type=str, help='Directory to watch')
    parser.add_argument('-f', '--frequency', default=5, type=float,
                        help='Update frequency in minutes')
    parser.add_argument('-a', '--age', default=0.5, type=float,
                        help='Minimum age of the checkpoints to delete in hours')
    parser.add_argument('-m', '--metric', default='validation/Accuracy', type=str,
                        help='Name of the metric being tracked')
    parser.add_argument('-t', '--metric_threshold', default=0.1, type=float,
                        help='Name of the metric being tracked')
    parser.add_argument('-w', '--wait_done', default=True, type=bool,
                        help='Wait for runs to be done before deleting')
    return vars(parser.parse_args())


def check_directory(path: Path,
                    age: float,
                    wait_done: bool = True,
                    metric: str = 'validation/Accuracy',
                    file_pattern: str = '.ckpt',
                    metric_threshold: float = -1,
                    level: int = 0):
    saved_size = 0
    for fn in path.iterdir():
        if fn.is_dir():
            saved_size += check_directory(fn, age,
                                          file_pattern=file_pattern,
                                          level=level + 1,
                                          metric=metric,
                                          wait_done=wait_done,
                                          metric_threshold=metric_threshold)
        elif str(fn).endswith(file_pattern):
            stats = fn.stat()
            ctime = max(stats.st_ctime, stats.st_mtime)
            fn_age = datetime.now() - datetime.fromtimestamp(ctime)
            fn_age = fn_age.total_seconds() / 3600.0
            fz_size = stats.st_size / (1024 ** 3)
            done, metric_value = attempt_reading_progress(fn, metric)
            delete_condition = fn_age > age
            delete_condition &= done is True if wait_done else True
            delete_condition &= metric_value < metric_threshold
            if delete_condition:
                saved_size += fz_size
                rich.print(
                    f"> [magenta]Deleting[/magenta]: [grey58]{fn}[/grey58]\n"
                    f"  | Age={fn_age:.1f} hours ({age:.1f}), "
                    f"{metric}={metric_value:.3f} ({metric_threshold:.3f}), "
                    f"done={done}")
                fn.unlink()
            else:
                rich.print(f"> [green]Keeping[/green]: [grey58]{fn}[/grey58]\n"
                           f"  | Age={fn_age:.1f} hours ({age:.1f}), "
                           f"{metric}={metric_value:.3f} ({metric_threshold:.3f}), "
                           f"done={done}")

    return saved_size


def attempt_reading_progress(fn, metric):
    done = metric_value = None
    for file in fn.parent.parent.iterdir():
        if str(file).endswith('progress.csv'):
            results = pd.read_csv(file)
            metric_value = results[metric].max()
            done = bool(results["done"].iloc[-1])
    return done, metric_value


if __name__ == '__main__':
    args = parse_args()
    rich.print(args)
    path = Path(args['directory'])
    assert path.is_dir()
    while True:
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        rich.print(f"=== {now_str} ===")
        saved_size = check_directory(path, args['age'],
                                     metric=args['metric'],
                                     wait_done=args['wait_done'],
                                     metric_threshold=args['metric_threshold'])

        rich.print(f">>> [green] Cleaned up {saved_size:.2f}GB")
        time.sleep(60 * args['frequency'])
