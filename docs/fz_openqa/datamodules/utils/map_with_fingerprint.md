# MapWithFingerprint

> Auto-generated documentation for [fz_openqa.datamodules.utils.map_with_fingerprint](blob/master/fz_openqa/datamodules/utils/map_with_fingerprint.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Utils](index.md#utils) / MapWithFingerprint
    - [MapWithFingerprint](#mapwithfingerprint)
        - [MapWithFingerprint().\_\_call\_\_](#mapwithfingerprint__call__)

## MapWithFingerprint

[[find in source code]](blob/master/fz_openqa/datamodules/utils/map_with_fingerprint.py#L22)

```python
class MapWithFingerprint():
    def __init__(
        pipe: Pipe,
        batched=True,
        cache_dir: str = None,
        _id: str = None,
        **map_kwargs: Any,
    ):
```

Make sure to set `new_fingerprint` to each split.

### MapWithFingerprint().\_\_call\_\_

[[find in source code]](blob/master/fz_openqa/datamodules/utils/map_with_fingerprint.py#L35)

```python
def __call__(dataset: HfDataset) -> HfDataset:
```

Apply the `pipe` to the `dataset` using deterministic fingerprints.

#### See also

- [HfDataset](typing.md#hfdataset)
