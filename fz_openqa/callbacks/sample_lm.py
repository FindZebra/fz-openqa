import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rich import print
from transformers import PreTrainedTokenizerFast


class SampleLanguageModel(Callback):
    def on_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        tokenizer: PreTrainedTokenizerFast = pl_module.tokenizer
        input_ids = pl_module.generate()

        self.display(input_ids, tokenizer)

    def display(self, input_ids, tokenizer):
        console_w, _ = shutil.get_terminal_size((100, 20))
        print(console_w * "-")
        for tokens in input_ids:
            print(tokenizer.decode(tokens).replace("[PAD]", ""))
        print(console_w * "-")
