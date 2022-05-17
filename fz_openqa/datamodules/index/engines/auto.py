from os import PathLike
from pathlib import Path
from typing import Any
from typing import Dict

from .base import IndexEngine
from .document_lookup import DocumentLookupEngine
from .es import ElasticsearchEngine
from .faiss import FaissEngine
from .token_faiss import FaissTokenEngine
from fz_openqa.utils.fingerprint import get_fingerprint

Engines = {
    "faiss": FaissEngine,
    "faiss_token": FaissTokenEngine,
    "doc_lookup": DocumentLookupEngine,
    "es": ElasticsearchEngine,
}


def AutoEngine(
    *,
    name: str,
    path: PathLike,
    config: Dict[str, Any] = None,
    set_unique_path: bool = False,
    **kwargs,
) -> IndexEngine:
    # get the constructor
    EngineCls = Engines[name]

    # get the deterministic config
    index_fingerprint = get_fingerprint(config)

    # get the index path
    if set_unique_path:
        path = Path(path) / f"{name}-{index_fingerprint}"

    return EngineCls(path=path, config=config, **kwargs)
