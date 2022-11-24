from typing import Dict

LEGACY_DROP_KEYS = [
    ("datamodule", "relevance_classifier"),
    ("datamodule", "builder", "relevance_classifier"),
    ("datamodule", "score_transform"),
    ("datamodule", "builder", "score_transform"),
]


def patch_engine(key, config: Dict) -> Dict:
    name_lookup = {"es": "elasticsearch", "faiss": "dense"}
    key_lookup = {
        "merge_max_size": "max_k",
        "metric_type": "faiss_metric",
    }
    # get the name of the index
    name = name_lookup.get(key, key)

    # populate the config with the known values and base attributes
    cfg = config.pop("config")
    cfg.update(config)
    cfg.update(
        {
            "query_field": "question",
            "index_field": "document",
            "auxiliary_field": "answer",
            "main_key": "text",
            "index_key": "row_idx",
            "group_key": "doc_idx",
            "score_key": "proposal_score",
        }
    )

    # override the name of some keys
    for k, v in key_lookup.items():
        if k in cfg:
            cfg[v] = cfg.pop(k)

    # special cases
    if "index_factory" in cfg:
        if "shard:" in cfg["index_factory"]:
            cfg["index_factory"] = cfg["index_factory"].replace("shard:", "")
            cfg["shard"] = True

    return {
        "name": name,
        "config": cfg,
    }


def patch_index_builder(config: Dict) -> Dict:
    dtype = config["dtype"]
    cache_dir = config["cache_dir"]
    engines = config["engines"]
    engines = {k: patch_engine(k, e_cfg) for k, e_cfg in engines.items()}
    id = config["id"]
    loader_kwargs = config["loader_kwargs"]
    cache_config = {
        "dtype": dtype,
        "cache_dir": cache_dir,
        "loader_kwargs": loader_kwargs,
    }
    return {
        "_target_": "fz_openqa.datamodules.builders.IndexBuilder",
        "engines": engines,
        "cache_dir": cache_dir,
        "index_cache_config": {**cache_config, "model_output_key": "_hd_"},
        "query_cache_config": {**cache_config, "model_output_key": "_hq_"},
        "id": id,
    }


def patch_legacy_config(config: Dict) -> Dict:
    """A small utility script to patch legacy configs to the new format."""
    for path_key in LEGACY_DROP_KEYS:
        cfg = config
        *path, key = path_key
        for p in path:
            cfg = cfg[p]
        cfg.pop(key, None)

    # patch the index builder
    if "index_builder" in config["datamodule"]:
        cfg = config["datamodule"]
        cfg["index_builder"] = patch_index_builder(cfg["index_builder"])

    return config
