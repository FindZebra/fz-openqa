# Predict Pipe

> Auto-generated documentation for [examples.predict_pipe](blob/master/examples/predict_pipe.py) module.

- [Fz-openqa](../README.md#fz-openqa-index) / [Modules](../MODULES.md#fz-openqa-modules) / [Examples](index.md#examples) / Predict Pipe
    - [run](#run)

#### Attributes

- `default_cache_dir` - define the default cache location: `Path(fz_openqa.__file__).parent.parent / 'cache'`

## run

[[find in source code]](blob/master/examples/predict_pipe.py#L57)

```python
@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name='script_config.yaml',
)
def run(config: DictConfig) -> None:
```

Load a corpus and index it using Faiss.
Then query the corpus using the 3 first corpus documents.

PytorchLightning's Trainer can be used to accelerate indexing.
Example, to index the whole corpus (~5min on `rhea`):

```bash
poetry run python examples/index_corpus_using_dense_retriever.py     trainer.strategy=dp trainer.gpus=8 +batch_size=2000 +num_workers=16
+n_samples=null +use_subset=False +num_proc=4

```
