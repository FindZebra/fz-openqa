from os import PathLike
from pathlib import Path
from typing import Any
from typing import Dict

import rich

from .base import IndexEngine
from .es import ElasticsearchEngine
from .faiss import FaissEngine
from .index_lookup import IndexLookupHandler
from fz_openqa.utils.fingerprint import get_fingerprint

Engines = {
    "faiss": FaissEngine,
    "lookup": IndexLookupHandler,
    "es": ElasticsearchEngine,
}


def AutoEngine(
    engine_name: str,
    *,
    path: PathLike,
    config: Dict[str, Any] = None,
    set_unique_path: bool = False,
    **kwargs,
) -> IndexEngine:
    # get the constructor
    EngineCls = Engines[engine_name]

    # get the deterministic config
    index_fingerprint = get_fingerprint(config)

    # get the index path
    if set_unique_path:
        path = Path(path) / f"{engine_name}-{index_fingerprint}"

    return EngineCls(path=path, config=config, **kwargs)
