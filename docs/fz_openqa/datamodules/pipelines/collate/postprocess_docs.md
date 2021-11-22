# Postprocess Docs

> Auto-generated documentation for [fz_openqa.datamodules.pipelines.collate.postprocess_docs](blob/master/fz_openqa/datamodules/pipelines/collate/postprocess_docs.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipelines](../index.md#pipelines) / [Collate](index.md#collate) / Postprocess Docs
    - [PostprocessPipe](#postprocesspipe)

## PostprocessPipe

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/collate/postprocess_docs.py#L20)

```python
class PostprocessPipe(BlockSequential):
    def __init__(
        relevance_classifier: RelevanceClassifier,
        n_retrieved_documents: int,
        n_select_documents: Optional[Union[int, Dict]],
        max_select_pos_docs: Optional[int],
        **kwargs,
    ):
```
