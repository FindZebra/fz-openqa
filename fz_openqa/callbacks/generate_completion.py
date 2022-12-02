from copy import copy
from copy import deepcopy
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import rich
import torch
from jinja2 import Template
from loguru import logger
from pydantic import BaseModel
from pytorch_lightning import Callback
from pytorch_lightning.utilities import move_data_to_device
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast
from warp_pipes import Batch
from warp_pipes.support.pretty import get_console_separator

import wandb
from fz_openqa.callbacks import templates
from fz_openqa.modeling import Model
from fz_openqa.modeling.modules import ReaderRetriever


class Completion(BaseModel):
    question: str
    answer: str
    completion: str
    id: str

    def rich_repr(self) -> str:
        canvas = ""
        canvas += f"Question ({self.id}): `[white]{self.question}[/white]`\n"
        canvas += f"Answer: `[green]{self.answer}[/green]`\n"
        canvas += f"Completion: `[red]{self.completion}[/red]`\n"
        return canvas


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
        self.tokenizer_left = deepcopy(tokenizer)
        self.tokenizer_left.padding_side = "left"
        self.tokenizer_left.truncation_side = "left"
        self.tokenizer_right = deepcopy(tokenizer)
        self.tokenizer_right.padding_side = "right"
        self.tokenizer_right.truncation_side = "right"
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

        # split the input into question and answer, so that we can generate the completion
        # from the questions and compare it to the answers
        questions, answers = self.separate_questions_and_answers(batch)
        answer_max_length = batch["input_ids"].size(1) - questions["input_ids"].size(1)

        # generate the completions
        gen_kwargs = {
            **self.generate_kwargs,
            "max_new_tokens": answer_max_length + 10,
            "pad_token_id": self.tokenizer_left.pad_token_id,
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
        completions = []
        for i, gen in enumerate(generated):
            q_ids = gen[:in_q_len]
            q_str = self.tokenizer_left.decode(q_ids, **decode_kwargs)
            a_str = self.tokenizer_left.decode(answers["input_ids"][i], **decode_kwargs)
            comp_str = self.tokenizer_left.decode(gen[in_q_len:], skip_special_tokens=False)
            comp = Completion(
                question=q_str,
                answer=a_str,
                completion=comp_str,
                id=str(i + 1),
            )
            completions.append(comp)

        canvas = f"{get_console_separator('.')}\n".join([c.rich_repr() for c in completions])
        canvas = f"{get_console_separator('=')}\n{canvas}{get_console_separator('=')}"

        # log
        if self.verbose:
            rich.print(canvas)
        with open(Path(templates.__file__).parent / "completions.html", "r") as f:
            html_template = Template(f.read())
        html = html_template.render(
            {
                "info": reader.config.model_type,
                "completions": completions,
            }
        )

        # save the HTML file
        output = Path() / "completions.html"
        with open(output, "w") as f:
            f.write(html)

        try:
            name = "completions/html"
            wandb.log({name: wandb.Html(html, inject=False)}, commit=False)
        except wandb.errors.Error as e:
            logger.warning(e)

    def separate_questions_and_answers(self, batch: Batch) -> Tuple[Batch, Batch]:
        """Remove the answer tokens from the input_ids and pad all the sequences left."""
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
        questions = self.tokenizer_left.pad(
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
        answers = self.tokenizer_right.pad(
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
