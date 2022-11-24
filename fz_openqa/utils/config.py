import os
import re
from copy import copy
from numbers import Number
from typing import List
from typing import Optional
from typing import Sequence

import omegaconf
import rich
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree

YAML_PATTERN = r"^.*\.yaml$"


class IntDict(dict):
    def __init__(self, **kwargs):
        super().__init__({int(k): v for k, v in kwargs.items()})


def null_constructor(*args, **kwargs):
    return None


def resolve_config_paths(config: DictConfig, path: str = "", excludes: List[str] = ["hydra"]):
    for k, v in ((k_, v_) for k_, v_ in config.items() if k_ not in excludes):
        if isinstance(v, DictConfig):
            resolve_config_paths(v, os.path.join(path, k))
        if isinstance(v, str) and re.findall(YAML_PATTERN, v):
            full_path = os.path.join(path, k, v)
            cfg = omegaconf.OmegaConf.load(full_path)
            config[k] = cfg


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Optional[Sequence[str]] = None,
    resolve: bool = True,
    exclude: Optional[Sequence[str]] = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        :param exclude:
    """

    style = "dim"
    tree = Tree(":gear: CONFIG", style=style, guide_style=style)
    if exclude is None:
        exclude = []

    fields = fields or list(config.keys())
    fields = list(filter(lambda x: x not in exclude, fields))

    with open_dict(config):
        base_config = {}
        for field in copy(fields):
            if isinstance(config.get(field), (bool, str, Number)):
                base_config[field] = config.get(field)
                fields.remove(field)
        config["__root__"] = base_config
    fields = ["__root__"] + fields

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            try:
                branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
            except Exception:
                pass

        branch.add(Syntax(branch_content, "yaml", indent_guides=True, word_wrap=True))

    rich.print(tree)
