# Colbert Data Constructor

> Auto-generated documentation for [fz_openqa.utils.colbert_data_constructor](blob/master/fz_openqa/utils/colbert_data_constructor.py) module.

Generate FZxMedQA Dataset

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Colbert Data Constructor
    - [getDocChunks](#getdocchunks)
    - [ingest_all](#ingest_all)

This script will generate the FZxMedQADataset using dataset files
with questions linked to FindZebra Corpus (using CUIs)
A running instance of ElasticSearch 7.13 must be running on localhost:9200

Run 'docker compose up' with the supplied docker-compose.yml file to start
two models (ElasticSearch and Kibana, both ver  7.13)

#### Attributes

- `out` - offical HuggingFace datastructure
  see: https://huggingface.co/docs/datasets/loading_datasets.html: `{'version': '0.0.1', 'data': []}`
- `es_res` - returning top n hits from es index based on question (query input): `es_search(ds_names[ds_id], ds[key]['question'], args.topn)`

## getDocChunks

[[find in source code]](blob/master/fz_openqa/utils/colbert_data_constructor.py#L63)

```python
def getDocChunks(article: str, chunkSize: int, stride: int):
```

takes in a FZ article as a string,
cuts the article into chunks, and
outputs a list of these chunks.

## ingest_all

[[find in source code]](blob/master/fz_openqa/utils/colbert_data_constructor.py#L88)

```python
def ingest_all(data: dict, dateset_name: str):
```

ingesting all articles in a dict dataset in sense of
chunks with the aim of searching through the index
to ranks chunks to a given input query.
