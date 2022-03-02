import logging
import sys
from pathlib import Path

import rich
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.pretty import pprint_batch

sys.path.append(str(Path(__file__).parent.parent))
rich.print(f"> root: {str(Path(__file__).parent.parent)}")

import datasets

from fz_openqa.modeling import Model
from fz_openqa.utils.config import print_config

logger = logging.getLogger(__name__)

import os

import hydra
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa import configs

rich.print(configs.__file__)


def int_div(a, *b):
    y = a
    for x in b:
        y = y // x
    return y


def int_mul(a, *b):
    y = int(a)
    for x in b:
        y *= int(x)
    return y


def int_max(a, *b):
    y = int(a)
    for x in b:
        y = max(x, y)
    return y


N_GPUS = torch.cuda.device_count()
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("int_mul", int_mul)
OmegaConf.register_new_resolver("int_div", int_div)
OmegaConf.register_new_resolver("int_max", int_max)
OmegaConf.register_new_resolver("n_gpus", lambda *_: N_GPUS)

from copy import deepcopy
from functools import partial

import rich
import torch

from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipelines.preprocessing import FormatAndTokenize
from fz_openqa.datamodules.pipes import TextFormatter, Parallel, Sequential, Apply, ConcatTextFields
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.nesting import Expand, ApplyAsFlatten
from fz_openqa.datamodules.utils.transformations import add_spec_token


class TestData():
    # Define dummy questions
    questions = ["Where is Paris?", "What is a Banana?"]

    # Define documents, including one for each question [#0, #4]
    documents = [
        [
            ["Paris is in France.",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. "
             "It contains algorithms that search in sets of vectors of any size, "
             "up to ones that possibly do not fit in RAM.",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
             ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. "
                "It contains algorithms that search in sets of vectors of any size, "
                "up to ones that possibly do not fit in RAM.",
                "Rockets are use to fly to space",
                "It also contains supporting code for evaluation and parameter tuning.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
            ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. "
                "It contains algorithms that search in sets of vectors of any size, "
                "up to ones that possibly do not fit in RAM.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "Bike is the vehicle of the future",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
            ],
        ],
        [
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. "
                "It contains algorithms that search in sets of vectors of any size, up to ones "
                "that possibly do not fit in RAM.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
                "Truck can carry tons of merchandises.",
            ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. "
                "It contains algorithms that search in sets of vectors of any size, up to ones "
                "that possibly do not fit in RAM.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "A banana is a fruit",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."
            ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. "
                "It contains algorithms that search in sets of vectors of any size, up to ones "
                "that possibly do not fit in RAM.",
                "Aircraft can fly across oceans.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."
            ]
        ]
    ]
    match_score = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]])
    doc_ids = torch.arange(0, match_score.numel(), 1).view(match_score.shape)

    # dummy answers
    answers = [["France", "Rocket", "Bike"], ["Truck", "Fruit", "Aircraft"]]
    answer_targets = torch.tensor([0, 1])

    def __init__(self, tokenizer: PreTrainedTokenizerFast, batch_size: int = 2):
        self.batch_size = batch_size
        self.n_documents = 4
        self.n_options = 3
        self.tokenizer = tokenizer

        # tokenize data
        self._batch = self._encode_data(self.data)

    def _encode_data(self, data: Batch) -> Batch:
        pipe = self.get_preprocessing_pipe()
        return pipe(data)

    @property
    def batch(self):
        return deepcopy(self._batch)

    @property
    def data(self):
        return {
            "question.text": self.questions,
            "document.text": self.documents,
            "answer.text": self.answers,
            "answer.target": self.answer_targets,
            "document.match_score": self.match_score,
            "document.row_idx": self.doc_ids
        }

    def get_preprocessing_pipe(self):
        # concat question and answer
        add_spec_tokens_pipe = Apply(
            {"question.text": partial(add_spec_token, self.tokenizer.sep_token)},
            element_wise=True
        )
        concat_qa = Sequential(
            add_spec_tokens_pipe,
            Expand(axis=1, n=self.n_options, update=True, input_filter=In(["question.text"])),
            ApplyAsFlatten(
                ConcatTextFields(keys=["answer.text", "question.text"],
                                 new_key="question.text"),
                level=1,
            ),
            input_filter=In(["question.text", "answer.text"]),
            update=True
        )

        # format and tokenize args
        args = {'text_formatter': TextFormatter(),
                'tokenizer': self.tokenizer,
                'max_length': 512,
                'add_special_tokens': True,
                'spec_tokens': None,
                }
        preprocess = Parallel(
            FormatAndTokenize(
                prefix="question.",
                shape=[-1, self.n_options],
                **args
            ),
            FormatAndTokenize(
                prefix="document.",
                shape=[-1, self.n_options, self.n_documents],
                **args
            ),
            update=True
        )
        collate = Parallel(
            CollateField("document",
                         tokenizer=self.tokenizer,
                         level=2,
                         to_tensor=["match_score", "row_idx"]
                         ),
            CollateField("question",
                         tokenizer=self.tokenizer,
                         level=1,
                         ),
            CollateField("answer",
                         include_only=["target"],
                         to_tensor=["target"],
                         level=0,
                         ),
        )
        return Sequential(concat_qa, preprocess, collate)


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)
    print_config(config, fields=['model'])

    # initialize the model
    model: Model = hydra.utils.instantiate(config.model, _recursive_=False)
    for k, v in model.named_parameters():
        if "reader" in k or "retriever" in k:
            rich.print(f"> {k}: {v.shape}, mean={v.mean():.2e}, std={v.std():.2e}")

    bert_params = {k: get_fingerprint(v) for k, v in model.named_parameters() if
                   "bert.encoder" in k}
    rich.print(f"> BERT fingerprint={get_fingerprint(bert_params)}")

    # instantiate the tokenizer + data
    # tokenizer = hydra.utils.instantiate(config.datamodule.tokenizer)
    # data = TestData(tokenizer, batch_size=2)
    # batch = data.batch

    paths = ["/scratch/valv/fz-openqa/runs/2022-03-01/14-37-56/batch=0.pt",
             "/scratch/valv/fz-openqa/runs/2022-03-01/14-37-56/batch=1.pt",
             "/scratch/valv/fz-openqa/runs/2022-03-01/14-37-56/batch=2.pt",
             "/scratch/valv/fz-openqa/runs/2022-03-01/14-37-56/batch=3.pt",
             ]
    batch = torch.load(paths[0], map_location=torch.device('cpu'))

    # test model forward
    pprint_batch(batch, "input")
    output = model(batch)
    pprint_batch(output, "output")
    for k, v in output.items():
        v = v.float()
        rich.print(f"> {k}: {v.shape}: {v.mean():.2e} ({v.std():.2e})")

    # test model evaluation
    step_output = model.training_step(batch, batch_idx=0)
    output = model.training_step_end(step_output, batch_idx=0)
    pprint_batch(output, "eval:output")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            v = v.float()
            rich.print(f"> {k}: {v.shape}: {v.mean():.2e} ({v.std():.2e})")
        else:
            rich.print(f"> {k}: {v}")


if __name__ == "__main__":
    run()
