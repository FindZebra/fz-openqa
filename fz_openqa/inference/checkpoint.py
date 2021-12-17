import logging
from pathlib import Path
from typing import Optional
from typing import Union

import datasets
from hydra._internal.instantiate._instantiate2 import _resolve_target
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast

from fz_openqa import configs
from fz_openqa.modeling import Model
from fz_openqa.utils.config import print_config
from fz_openqa.utils.datastruct import PathLike

logger = logging.getLogger(__name__)


def get_drive_url(url):
    base_url = "https://drive.google.com/uc?id="
    split_url = url.split("/")
    return base_url + split_url[5]


def download_asset_from_gdrive(
    url: str, cache_dir: Optional[str] = None, extract: bool = False
) -> str:
    dl_manager = datasets.utils.download_manager.DownloadManager(
        dataset_name="fz_ner_annotator", data_dir=cache_dir
    )
    if extract:
        return dl_manager.download_and_extract(get_drive_url(url))
    else:
        return dl_manager.download(get_drive_url(url))


def maybe_download_weights(checkpoint: str, cache_dir: Optional[str] = None) -> str:
    is_url = isinstance(checkpoint, str) and checkpoint.startswith("https://")
    if is_url:
        logger.info(f"Using weights from {checkpoint}")
        checkpoint = download_asset_from_gdrive(checkpoint, cache_dir=cache_dir, extract=True)
    return checkpoint


class CheckpointLoader:
    _tokenizer: Optional[PreTrainedTokenizerFast] = None

    def __init__(
        self,
        checkpoint_dir: str,
        override: Optional[DictConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        checkpoint_dir = maybe_download_weights(checkpoint_dir, cache_dir=cache_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        assert self.checkpoint_dir.is_dir(), f"{checkpoint_dir} is not a directory"

        # if the directory contains only a single directory, assume it's the checkpoint directory
        files = [p for p in self.checkpoint_dir.iterdir() if ".DS_Store" not in p.name]
        if all(p.is_dir() for p in files) and len([p for p in files]) == 1:
            self.checkpoint_dir = files[0]
        logger.info(f"Loading checkpoint from {self.checkpoint_dir}")

        # load the checkpoint config
        self.config = OmegaConf.load(self.checkpoint_dir / "config.yaml")
        if override is not None:
            logger.info(f"Overriding config with keys {list(override.keys())}")
            if "override_ops" in override.keys():
                self._apply_override_ops(override.override_ops)
            self.config = OmegaConf.merge(self.config, override)

    def _apply_override_ops(self, override_config: DictConfig):

        delete_keys = override_config.get("delete", None)
        if delete_keys is not None:
            for key in delete_keys:
                self.config.pop(key, None)

    def print_config(self, **kwargs):
        print_config(self.config, resolve=False, **kwargs)

    def model_checkpoint(self, last=False) -> Union[None, Path]:
        checkpoints = (self.checkpoint_dir / "checkpoints").iterdir()
        checkpoints = filter(lambda x: x.suffix == ".ckpt", checkpoints)
        if last:
            checkpoints = filter(lambda x: "last" in x.name, checkpoints)
        else:
            checkpoints = filter(lambda x: "last" not in x.name, checkpoints)

        try:
            return next(iter(checkpoints))
        except StopIteration:
            return None

    def load_tokenizer(self):
        return instantiate(self.config.datamodule.tokenizer)

    def instantiate(self, path: str, **kwargs):
        cfg = self.config
        for p in path.split("."):
            cfg = getattr(cfg, p)

        return instantiate(cfg, **kwargs)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    def load_bert(self):
        return instantiate(self.config.model.bert)

    def load_model(self, last=False, **kwargs) -> Model:
        logger.info(f"Instantiating model <{self.config.model._target_}>")

        path = self.model_checkpoint(last=last)
        if path is not None:
            logger.info(f"Loading model from checkpoint: {path}")
            # need to override the saved `tokenizer` and `bert` hyperparameters
            # so `sys.cache_dir` can be overridden
            cls = _resolve_target(self.config.model._target_)
            model = cls.load_from_checkpoint(
                path,
                tokenizer=OmegaConf.to_object(self.config.datamodule.tokenizer),
                bert=OmegaConf.to_object(self.config.model.bert),
                **kwargs,
            )
        else:
            logger.warning("No checkpoint found. Initializing model without checkpoint.")
            model = instantiate(self.config.model, _recursive_=False)

        return model
