from __future__ import annotations

import re

from fz_openqa.utils.datastruct import Batch


def slice_batch(batch: Batch, i: int | slice) -> Batch:
    return {k: v[i] for k, v in batch.items()}


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
