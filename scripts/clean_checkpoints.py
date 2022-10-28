# find . -type d -path "./*" -mmin +120 -exec rm -rf {} \;
import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from typing import Iterable

import numpy as np
import pandas as pd
import rich
from rich.console import Console
from rich.table import Table

MIN_MAX_AGE = 6  # raise an error if trying to delete runs younger than that


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--directory', default="/scratch/valv/raytune/",
                        type=str, help='Directory to watch')
    parser.add_argument('-f', '--frequency', default=5, type=float,
                        help='Update frequency in minutes')
    parser.add_argument('-a', '--max_age', default=12, type=float,
                        help='Minimum age of the checkpoints to delete in hours')
    parser.add_argument('-m', '--metric', default='validation/F1', type=str,
                        help='Name of the metric being tracked')
    parser.add_argument('-t', '--total', default=20, type=float,
                        help='Maximum of finsihed runs to keep')
    return vars(parser.parse_args())


def scan_directory(path: Path,
                   metric: str = 'validation/F1',
                   file_pattern: str = '.ckpt',
                   level: int = 0) -> Iterable[Dict[str, Any]]:
    for fn in path.iterdir():
        if fn.is_dir():
            for exp in scan_directory(fn,
                                      file_pattern=file_pattern,
                                      level=level + 1,
                                      metric=metric):
                yield exp

        elif str(fn).endswith(file_pattern):
            stats = fn.stat()
            ctime = max(stats.st_ctime, stats.st_mtime)
            fn_age = datetime.now() - datetime.fromtimestamp(ctime)
            fn_age = fn_age.total_seconds() / 3600.0
            fz_size = stats.st_size / (1024 ** 3)
            done, fn_metric = attempt_reading_progress(fn, metric)

            yield {'fn': fn,
                   'age': fn_age,
                   'done': done,
                   'metric': fn_metric,
                   'size': fz_size}


def attempt_reading_progress(fn, metric):
    done = metric_value = None
    for file in fn.parent.parent.iterdir():
        if str(file).endswith('progress.csv'):
            results = pd.read_csv(file)
            metric_value = results[metric].max()
            done = bool(results["done"].iloc[-1])
    return done, metric_value


def display_and_delete(runs: pd.DataFrame, console: Console, *, root: str,
                       metric_name: str, max_age: float, max_rank: int):
    assert max_age > MIN_MAX_AGE, f"{max_age}h-old runs are too young to die. That's probably a mistake."
    table = init_table(metric_name, root)
    counter = 0
    garbage = []

    # add the color column
    runs['color'] = get_color(runs, "metric")

    # process running runs
    running_runs = runs[
        (runs["done"] == False) & (runs["age"] <= max_age)].sort_values(
        "metric",
        ascending=False).reset_index()
    for idx, row in running_runs.iterrows():
        counter += 1
        keep_this = True
        status = "[cyan]running[/cyan]"
        add_row(idx, keep_this, row, status,
                table, len(running_runs), root)
        if not keep_this:
            garbage += [row]

    # process finished runs
    finished_runs = runs[runs["done"] == True].sort_values("metric",
                                                           ascending=False).reset_index()
    for idx, row in finished_runs.iterrows():
        counter += 1
        status = "[green]completed[/green]"
        keep_this = idx < max_rank
        add_row(idx, keep_this, row, status, table, len(finished_runs), root)
        if not keep_this:
            garbage += [row]

    # process stale runs
    stale_runs = runs[
        (runs["done"] == False) & (runs["age"] > max_age)].sort_values("metric",
                                                                       ascending=False).reset_index()
    for idx, row in stale_runs.iterrows():
        counter += 1
        keep_this = False
        status = "[magenta]stale[/magenta]"
        add_row(idx, keep_this, row, status, table, len(stale_runs), root,
                stale=True)
        if not keep_this:
            garbage += [row]

    # display the table
    console.print(table)
    assert counter == len(runs), "Some runs were not displayed in the table"

    # delete runs
    saved_space = 0
    count = 0
    for row in garbage:
        count += 1
        saved_space += row['size']
        os.remove(row['fn'])

    check_symbol = u'\N{check mark}'
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rich.print(
        f"  {check_symbol} {now_str} [green]Cleaned up {saved_space:.2f}GB, {count} runs[/green]")


def add_row(idx, keep_this, row, status, table, length, root, stale=False):
    keep_this_str = f"[bold green]no[/bold green]" if keep_this else f"[bold magenta]yes[/bold magenta]"
    age_str = f"[magenta]{row['age']:.1f}[/magenta]" if stale else f"{row['age']:.1f}"
    path = str(row['fn']).replace(str(root), '...')
    table.add_row(f"{idx}",
                  f"[{row['color']}]{row['metric']:.3f}",
                  age_str,
                  status,
                  keep_this_str,
                  f"[grey58]{path}[/grey58]",
                  end_section=idx == length - 1)


def init_table(metric_name, root):
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    table = Table(title=f"=== {now_str} | root={root}")
    table.add_column("rank", justify="center", style="", no_wrap=True)
    table.add_column(metric_name, justify="center", style="cyan", no_wrap=True)
    table.add_column("age (h)", justify="center", style="", no_wrap=True)
    table.add_column("status", justify="center", style="", no_wrap=True)
    table.add_column("delete", justify="center", style="", no_wrap=True)
    table.add_column("path", justify="left", style="")
    return table


def get_color(df, column_name):
    colors = [f"#{x}" for x in
              ["7400b8", "6930c3", "5e60ce", "5390d9", "4ea8de", "48bfe3",
               "56cfe1", "64dfdf", "72efdd", "80ffdb"]]
    colors = np.array(colors[::-1])
    v = df[column_name].values
    v = ((v - v.min()) / (v.max() - v.min()) * (len(colors) - 1)).astype(
        np.int16)
    return pd.Series(colors[v])


if __name__ == '__main__':
    args = parse_args()
    console = Console()
    rich.print({k:v for k,v in args.items()})
    path = Path(args['directory'])
    assert path.is_dir()
    with console.status(f"Checking {args['directory']} "
                        f"every {args['frequency']} minutes ...") as status:
        # scan directory
        runs = scan_directory(path, metric=args['metric'])

        # cast to a Dataframe and sort by status
        runs = pd.DataFrame(runs)
        display_and_delete(runs, console, root=path, metric_name=args['metric'],
                           max_age=args['max_age'], max_rank=args['total'])

        # rich.print(f">>> [green] Cleaned up {saved_size:.2f}GB")
        time.sleep(60 * args['frequency'])
