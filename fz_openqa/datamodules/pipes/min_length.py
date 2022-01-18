from fz_openqa.utils.datastruct import Eg


class MinLength:
    def __init__(self, key: str, min_length: int):
        self.key = key
        self.min_length = min_length

    def __call__(self, row: Eg, **kwargs) -> bool:
        x = row[self.key]
        if isinstance(x, str):
            return len(x) >= self.min_length
        elif isinstance(x, list):
            return all(len(y) >= self.min_length for y in x)
        else:
            raise TypeError(f"{self.key} is not a string or list")
