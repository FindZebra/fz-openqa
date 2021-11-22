import logging
import os
from pathlib import Path
from typing import Any
from typing import Optional

import datasets
import hydra
import pytorch_lightning
import pytorch_lightning as pl
import rich
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from transformers import AutoModel

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.builders import FzCorpusBuilder
from fz_openqa.datamodules.builders import FZxMedQaCorpusBuilder
from fz_openqa.datamodules.builders import MedQaBuilder
from fz_openqa.datamodules.builders import MedQaCorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import FaissIndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config
from fz_openqa.utils.datastruct import Batch

logger = logging.getLogger(__name__)

DEFAULT_CKPT = "https://drive.google.com/file/d/17XDASu1JYGndCFWNW3zCZGKD2woJuDIg/view?usp=sharing"
default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


class ZeroShot(pl.LightningModule):
    def __init__(self, bert_id: str = "dmis-lab/biobert-base-cased-v1.2", **kwargs):
        super(ZeroShot, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_id)

    def forward(self, batch: Batch, **kwargs) -> Any:
        output = {}
        key_map = {"document": "_hd_", "question": "_hq_"}
        for prefix in ["document", "question"]:
            if any(prefix in k for k in batch.keys()):
                input_ids = batch[f"{prefix}.input_ids"]
                attention_mask = batch[f"{prefix}.attention_mask"]
                h = self.bert(input_ids, attention_mask).last_hidden_state
                output[key_map[prefix]] = h[:, 0, :]
        return output


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config):
    """Load the OpenQA dataset mapped using a pre-trained model.

    On the cluster, run:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python examples/load_mapped_medqa_faiss.py
    sys=titan trainer.strategy=dp trainer.gpus=8 +batch_size=2000 +num_workers=10 +use_subset=False
    ```
    """
    print_config(config)
    # set the context
    datasets.set_caching_enabled(True)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)
    try:
        cache_dir = config.sys.cache_dir
    except Exception:
        cache_dir = default_cache_dir

    # load model
    if config.get("zero_shot", False):
        model = ZeroShot()
    else:
        loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
        if config.get("verbose", False):
            loader.print_config()
        model = loader.load_model(last=config.get("last", False))
    model.eval()
    model.freeze()

    # Init Lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer.get('_target_', None)}>")
    trainer: Optional[Trainer] = instantiate(config.trainer)
    if isinstance(trainer, (dict, DictConfig)):
        logger.info("No Trainer was provided. PyTorch Lightning acceleration cannot be used.")
        trainer = None

    # tokenizer and text formatter
    tokenizer = init_pretrained_tokenizer(
        pretrained_model_name_or_path="dmis-lab/biobert-base-cased-v1.2"
    )
    text_formatter = TextFormatter(lowercase=True)

    # define the medqa builder
    dataset_builder = MedQaBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("use_subset", True),
        cache_dir=cache_dir,
        num_proc=4,
    )
    dataset_builder.subset_size = [200, 50, 50]

    # define the corpus builder
    corpus_builder = MedQaCorpusBuilder(
        tokenizer=tokenizer,
        text_formatter=text_formatter,
        use_subset=config.get("corpus_subset", False),
        cache_dir=cache_dir,
        num_proc=4,
    )

    # define the index builder
    index_builder = FaissIndexBuilder(
        model=model,
        trainer=trainer,
        model_output_keys=["_hd_", "_hq_"],
        collate_pipe=corpus_builder.get_collate_pipe(),
        loader_kwargs={
            "batch_size": config.get("batch_size", 10),
            "num_workers": config.get("num_workers", 4),
            "pin_memory": config.get("pin_memory", True),
        },
        cache_dir=cache_dir,
    )

    # define the OpenQA builder
    builder = OpenQaBuilder(
        dataset_builder=dataset_builder,
        corpus_builder=corpus_builder,
        index_builder=index_builder,
        relevance_classifier=ExactMatch(interpretable=True),
        n_retrieved_documents=1000,
        n_documents=10,
        max_pos_docs=1,
        filter_unmatched=True,
        num_proc=4,
        batch_size=100,
    )

    # define the data module
    dm = DataModule(builder=builder)

    # preprocess the data
    dm.prepare_data()
    dm.setup()
    dm.display_samples(n_samples=3)

    # access dataset
    rich.print(dm.dataset)

    # sample a batch
    # _ = next(iter(dm.train_dataloader()))


if __name__ == "__main__":
    run()
