import json
import re
import shutil
from pathlib import Path

import rich
from loguru import logger
from tqdm import tqdm

greek_alphabet = "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
latin_alphabet = "AaBbGgDdEeZzHhJjIiKkLlMmNnXxOoPpRrSssTtUuFfQqYyWw"
greek2latin = str.maketrans(greek_alphabet, latin_alphabet)


def clean_title(title):
    title = title.lower()
    title = title.translate(greek2latin)
    for a, b in [
        ("+", "plus"),
        ("-", "minus"),
        ("$", "dollar"),
        ("%", "percent"),
        ("&", "and"),
        ("€", "euro"),
    ]:
        title = title.replace(a, b)
    title = re.sub(r"[^a-z0-9]", "-", title)
    return title


def clean_text(text):
    lines = text.split("\n")
    for stop_pattern in ["See also ", "See also", "References ", "References", "External links"]:
        if stop_pattern in lines:
            stop_idx = lines.index(stop_pattern)
            lines = lines[:stop_idx]
    return "\n".join(lines)


def cleanup(input_file, output_dir):
    logger.info(f"Cleaning up {input_file} to {output_dir}")
    assert input_file.is_dir(), "File not found"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    n = 0
    for p in tqdm(input_file.iterdir(), unit="articles"):
        if not p.suffix == ".txt":
            continue
        with open(p, "r") as f_in:
            title = f_in.readline().strip()
            text = f_in.read()
            cleaned_title = clean_title(title)
            text = clean_text(text)
            with open(output_dir / f"{cleaned_title}.txt", "w") as f:
                f.write(f"{title}\n{text}")
            n += 1

    logger.info(f"{n} articles written to {output_dir}, " f"{n / 6487149:.2%} of Wikipedia")


if __name__ == "__main__":
    in_directory = Path("/Users/valv/Downloads/")
    out_directory = Path("/Users/valv/Downloads/")
    names = [
        "med_x_wiki_corpus_us",
        "med_x_wiki_corpus_tw",
        "med_x_wiki_corpus_us_tw",
    ]

    for name in names:
        input_file = in_directory / f"{name}_raw"
        out_file = out_directory / f"{name}"
        cleanup(input_file, out_file)
