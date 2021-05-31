from torch import nn
from transformers import AutoModel


class LanguageModelPrior(nn.Module):
    def __init__(self, *, pretrained_model_name: str, transformers_cache: str = None):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(
            pretrained_model_name, cache_dir=transformers_cache
        )
        self.vocab_size = self.transformer.config.vocab_size
