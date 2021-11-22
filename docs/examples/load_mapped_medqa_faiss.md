# Load Mapped Medqa Faiss

> Auto-generated documentation for [examples.load_mapped_medqa_faiss](blob/master/examples/load_mapped_medqa_faiss.py) module.

- [Fz-openqa](../README.md#fz-openqa-index) / [Modules](../MODULES.md#fz-openqa-modules) / [Examples](index.md#examples) / Load Mapped Medqa Faiss
    - [ZeroShot](#zeroshot)
        - [ZeroShot().forward](#zeroshotforward)
    - [run](#run)

## ZeroShot

[[find in source code]](blob/master/examples/load_mapped_medqa_faiss.py#L45)

```python
class ZeroShot(pl.LightningModule):
    def __init__(bert_id: str = 'dmis-lab/biobert-base-cased-v1.2', **kwargs):
```

### ZeroShot().forward

[[find in source code]](blob/master/examples/load_mapped_medqa_faiss.py#L50)

```python
def forward(batch: Batch, **kwargs) -> Any:
```

#### See also

- [Batch](../fz_openqa/utils/datastruct.md#batch)

## run

[[find in source code]](blob/master/examples/load_mapped_medqa_faiss.py#L62)

```python
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name='script_config.yaml',
)
def run(config):
```

Load the OpenQA dataset mapped using a pre-trained model.

On the cluster, run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python examples/load_mapped_medqa_faiss.py
sys=titan trainer.strategy=dp trainer.gpus=8 +batch_size=2000 +num_workers=10 +use_subset=False
```
