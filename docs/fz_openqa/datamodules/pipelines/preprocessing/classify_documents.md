# ClassifyDocuments

> Auto-generated documentation for [fz_openqa.datamodules.pipelines.preprocessing.classify_documents](blob/master/fz_openqa/datamodules/pipelines/preprocessing/classify_documents.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipelines](../index.md#pipelines) / [Preprocessing](index.md#preprocessing) / ClassifyDocuments
    - [ClassifyDocuments](#classifydocuments)

## ClassifyDocuments

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/preprocessing/classify_documents.py#L11)

```python
class ClassifyDocuments(Sequential):
    def __init__(
        corpus_dataset: Dataset,
        relevance_classifier: RelevanceClassifier,
    ):
```
