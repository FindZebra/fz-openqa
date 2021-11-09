import os
import re
from typing import List
from typing import Optional
from typing import Sequence

import omegaconf
import rich.syntax
import rich.tree
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

YAML_PATTERN = r"^.*\.yaml$"


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
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    fields = fields or config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            try:
                branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
            except Exception:
                pass

        branch.add(rich.syntax.Syntax(branch_content, "yaml", indent_guides=True, word_wrap=True))

    rich.print(tree)
