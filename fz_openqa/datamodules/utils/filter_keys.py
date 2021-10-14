from typing import List


class FilterKey:
    def __call__(self, key: str) -> bool:
        raise NotImplementedError


class KeyIn(FilterKey):
    """check if the key is in the allowed_keys"""

    def __init__(self, allowed_keys: List[str]):
        self.allowed_keys = allowed_keys

    def __call__(self, key: str) -> bool:
        return key in self.allowed_keys


class KeyWithPrefix(FilterKey):
    """check if the key starts with a given prefix"""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, key: str) -> bool:
        return str(key).startswith(self.prefix)
