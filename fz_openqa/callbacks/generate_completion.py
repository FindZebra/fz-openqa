from copy import copy
from typing import Dict
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import rich
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import move_data_to_device
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast
from warp_pipes import Batch
from warp_pipes import pprint_batch
from warp_pipes.support.pretty import get_console_separator

from fz_openqa.modeling import Model
from fz_openqa.modeling.modules import ReaderRetriever


class GenerateCompletionsCallback(Callback):
    """
    Log model predictions as plain text.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        log_dir: str = None,
        generate_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        n_samples: int = 10,
    ):
        super(GenerateCompletionsCallback, self).__init__()
        self.tokenizer = copy(tokenizer)
        if self.tokenizer.padding_side != "left":
            raise ValueError(
                f"Tokenizer must have padding_side='left'. "
                f"Found padding_side='{self.tokenizer.padding_side}'"
            )
        self.log_dir = log_dir
        self.verbose = verbose
        self.n_samples = n_samples

        generate_kwargs_ = {"temperature": 0.0, "max_new_tokens": 100, "num_beams": 1}
        if generate_kwargs is not None:
            generate_kwargs_.update(generate_kwargs)
        self.generate_kwargs = generate_kwargs_

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx == 0 and dataloader_idx == 0:
            return self.on_first_validation_batch_start(trainer, pl_module, batch)

    @torch.no_grad()
    def on_first_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Batch,
    ) -> None:
        reader = self._get_reader(pl_module)
        if reader is None:
            rich.print(f" Model {type(pl_module.module)} is not compatible with {type(self)}")
            return

        # fetch, reshape and clone the input data
        batch = {k.replace("lm.", ""): v for k, v in batch.items() if k.startswith("lm.")}
        batch = {k: v.view(-1, v.size(-1))[: self.n_samples] for k, v in batch.items()}
        pprint_batch(batch, "GenerateCompletionsCallback")

        questions, answers = self.separate_questions_and_answers(batch)
        answer_max_length = batch["input_ids"].size(1) - questions["input_ids"].size(1)

        # generate completions
        gen_kwargs = {
            **self.generate_kwargs,
            "max_new_tokens": answer_max_length + 10,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        questions = move_data_to_device(questions, pl_module.device)
        generated = reader.generate(
            inputs=questions["input_ids"],
            attention_mask=questions["attention_mask"],
            **gen_kwargs,
        )

        # decode the completions and format them
        in_q_len = questions["input_ids"].size(1)
        decode_kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}
        completion_reprs = []
        for i, gen in enumerate(generated):
            q_str = self.tokenizer.decode(gen[:in_q_len], **decode_kwargs)
            a_str = self.tokenizer.decode(gen[in_q_len:], **decode_kwargs)
            exp_str = self.tokenizer.decode(answers["input_ids"][i], **decode_kwargs)
            repr = f"Question: #{i + 1}: `[white]{q_str}[/white]`\n"
            repr += f"Completion: `[green]{a_str}[/green]`\n"
            repr += f"Expected: `[red]{exp_str}[/red]`\n"
            completion_reprs.append(repr)

        canvas = f"{get_console_separator('.')}\n".join(completion_reprs)
        canvas = f"{get_console_separator('=')}\n{canvas}{get_console_separator('=')}"

        # print
        if self.verbose:
            rich.print(canvas)

        # log
        # TODO: log to wandb

    def separate_questions_and_answers(self, batch: Batch) -> Tuple[Batch, Batch]:
        """Remove the answer tokens from the input_ids and pad all the sequences left."""
        assert self.tokenizer.padding_side == "left"
        input_ids = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"].clone()
        token_type_ids = batch["token_type_ids"].clone()

        # find the first and last tokens of the question
        ids = torch.arange(input_ids.size(1), device=input_ids.device)[None, :].expand_as(input_ids)
        mask = token_type_ids.clone()
        mask[mask == 1] = -1e9
        question_end_ids = 1 + (ids + mask).max(dim=1).values
        is_pad = attention_mask == 0
        is_pad[is_pad] = 1e9
        question_start_ids = 1 + (ids + is_pad).min(dim=1).values

        # fetch the question tokens and pad
        questions = self.tokenizer.pad(
            {
                "input_ids": [
                    input_ids[i, question_start_ids[i] : question_end_ids[i]]
                    for i in range(input_ids.size(0))
                ],
                "attention_mask": [
                    attention_mask[i, question_start_ids[i] : question_end_ids[i]]
                    for i in range(input_ids.size(0))
                ],
                "token_type_ids": [
                    token_type_ids[i, question_start_ids[i] : question_end_ids[i]]
                    for i in range(input_ids.size(0))
                ],
            },
            padding="longest",
        )

        # fetch the answer tokens and pad
        answers = self.tokenizer.pad(
            {
                "input_ids": [
                    input_ids[i, question_end_ids[i] :] for i in range(input_ids.size(0))
                ],
                "attention_mask": [
                    attention_mask[i, question_end_ids[i] :] for i in range(input_ids.size(0))
                ],
                "token_type_ids": [
                    token_type_ids[i, question_end_ids[i] :] for i in range(input_ids.size(0))
                ],
            },
            padding="longest",
        )

        return dict(questions), dict(answers)

    def _get_reader(self, pl_module: "pl.LightningModule") -> Optional[PreTrainedModel]:

        if not isinstance(pl_module, Model):
            return None

        if not isinstance(pl_module.module, ReaderRetriever):
            return None

        if hasattr(pl_module.module.reader, "generate"):
            return pl_module.module.reader

        return None
