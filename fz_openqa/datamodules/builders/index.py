from copy import copy

from datasets import Dataset
from omegaconf import DictConfig
from warp_pipes import Index
from warp_pipes.support.caching import CacheConfig


class IndexBuilder:
    cls: Index.__class__ = Index

    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(
        self,
        corpus: Dataset,
        *,
        index_collate_fn=None,
        query_collate_fn=None,
        trainer=None,
        **kwargs,
    ) -> Index:
        params = copy(self.params)
        params.update(kwargs)

        # set the collate functions and trainer dynamically
        if "index_cache_config" in params:
            if index_collate_fn is not None:
                cache_config = params["index_cache_config"]
                self._inject_config_attr(cache_config, "collate_fn", index_collate_fn)
                if trainer is not None:
                    self._inject_config_attr(cache_config, "trainer", trainer)
        else:
            raise ValueError(f"index_cache_config must be provided. Found {params.keys()}")

        if "query_cache_config" in params:
            if query_collate_fn is not None:
                cache_config = params["query_cache_config"]
                self._inject_config_attr(cache_config, "collate_fn", query_collate_fn)
                if trainer is not None:
                    self._inject_config_attr(cache_config, "trainer", trainer)
        else:
            raise ValueError(f"query_cache_config must be provided. Found {params.keys()}")

        return self.cls(corpus, **params)

    def _inject_config_attr(self, cache_config, key, value):
        if isinstance(cache_config, (dict, DictConfig)):
            cache_config[key] = value
        elif isinstance(cache_config, CacheConfig):
            setattr(cache_config, f"{key}_", value)
        else:
            raise TypeError(f"Unknown cache_config type: {type(cache_config)}")

    def __repr__(self):
        return f"{self.__class__.__name__}(engines={self.params['engines']})"
