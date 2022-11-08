from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import yaml
from datasets import DownloadManager
from hydra._internal.instantiate._instantiate2 import _resolve_target
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from packaging.version import parse as parse_version
from transformers import PreTrainedTokenizerFast

from fz_openqa.inference.checkpoint_legacy import patch_legacy_config
from fz_openqa.modeling import Model
from fz_openqa.utils.config import print_config


def get_drive_url(url):
    base_url = "https://drive.google.com/uc?id="
    split_url = url.split("/")
    return base_url + split_url[5]


def download_asset(url: str, cache_dir: Optional[str] = None, extract: bool = False) -> str:
    dl_manager = DownloadManager(dataset_name="fz_openqa_assets", data_dir=cache_dir)
    if extract:
        if "drive.google" in url:
            url = get_drive_url(url)
        return dl_manager.download_and_extract(url)
    else:
        return dl_manager.download(get_drive_url(url))


def maybe_download_weights(checkpoint: str, cache_dir: Optional[str] = None) -> str:
    is_url = isinstance(checkpoint, str) and checkpoint.startswith("https://")
    if is_url:
        logger.info(f"Using weights from {checkpoint}")
        checkpoint = download_asset(checkpoint, cache_dir=cache_dir, extract=True)
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
        self.config = self._load_config(self.checkpoint_dir / "config.yaml")

        if override is not None:
            logger.info(f"Overriding config with keys {list(override.keys())}")
            if "override_ops" in override.keys():
                self._apply_override_ops(override.override_ops)
            self.config = OmegaConf.merge(self.config, override)

    def _load_config(self, path: Union[str, Path]):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        code_version = parse_version(config["code_version"])
        if code_version < parse_version("0.2.0"):
            logger.warning(
                f"Code version {code_version} is older than 0.2.0. Patching legacy config."
            )
            config = patch_legacy_config(config)

        # load the checkpoint config
        config = OmegaConf.create(config)
        return config

    def _apply_override_ops(self, override_config: DictConfig):
        delete_keys = override_config.get("delete", None)
        if delete_keys is not None:
            for key in delete_keys:
                self.config.pop(key, None)

    def print_config(self, **kwargs):
        print_config(self.config, resolve=False, **kwargs)

    def model_checkpoint_paths(self, match: Optional[str] = None) -> List[Path]:
        checkpoints = (self.checkpoint_dir / "checkpoints").iterdir()
        checkpoints = filter(lambda x: x.suffix == ".ckpt", checkpoints)
        if match is not None:
            checkpoints = filter(lambda x: match in x.name, checkpoints)
        return list(checkpoints)

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
        return instantiate(self.config.model.backbone)

    def load_model(self, checkpoint_type="last", zero_shot: bool = False, **kwargs) -> Model:
        logger.info(f"Instantiating model <{self.config.model._target_}>")

        # find checkpoint files
        paths = self.model_checkpoint_paths(match=checkpoint_type)
        if len(paths) == 0 or zero_shot:
            if zero_shot:
                logger.info("Zero-shot. Initializing model without checkpoint.")
            else:
                logger.warning("No checkpoint found. Initializing model without checkpoint.")

            model: Model = instantiate(self.config.model, _recursive_=False, **kwargs)
            return model

        # select one checkpoint path
        if len(paths) > 1:
            logger.warning(
                f"Found multiple checkpoints: {[p.name for p in paths]}. "
                f"Using the last one in the list."
            )
        path = paths[-1]

        # load checkpoint state_dict
        logger.info(f"Loading model from checkpoint: {path}")
        cls: Model.__class__ = _resolve_target(self.config.model._target_, full_key=None)
        model = cls.load_from_checkpoint(checkpoint_path=str(path), map_location="cpu")
        return model
