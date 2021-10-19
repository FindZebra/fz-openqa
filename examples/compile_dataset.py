import json

import datasets
import rich
import torch

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env

datasets.set_caching_enabled(True)
setup_safe_env()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

text_formatter = TextFormatter(lowercase=True)

# load the corpus object
corpus = MedQaCorpusDataModule(
    tokenizer=tokenizer,
    text_formatter=text_formatter,
    index=ElasticSearchIndex(
        index_key="idx",
        text_key="document.text",
        query_key="question.text",
        num_proc=4,
        filter_mode=None,
    ),
    verbose=False,
    num_proc=4,
    use_subset=False,
)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    text_formatter=text_formatter,
    train_batch_size=100,
    num_proc=4,
    num_workers=4,
    use_subset=True,
    verbose=True,
    corpus=corpus,
    # retrieve 100 documents for each question
    n_retrieved_documents=100,
    # keep only one positive doc
    max_pos_docs=10,
    # keep only 10 docs (1 pos + 9 neg)
    n_documents=None,
    # simple exact match
    relevance_classifier=ExactMatch(interpretable=True),
)

# prepare both the QA dataset and the corpus
dm.subset_size = [100, 50, 50]
dm.prepare_data()
dm.setup()

print(get_separator())
dm.build_index()
rich.print("[green]>> index is built.")
print(get_separator())

# Compile the dataset
# ExactMatch: full dataset, num_proc=4, 1000 docs, bs=10: ~8s/batch, phoebe.compute.dtu.dk
# >  - train: 3473 (34.12%)
# >  - validation: 474 (37.26%)
# >  - test: 450 (35.35%)
# SciSpacyMatch: full dataset, num_proc=4, 1000 docs, bs=10: ~75s/batch, phoebe.compute.dtu.dk
# >  - train: 7605 (74.72%)
# >  - validation: 967 (76.02%)
# >  - test: 954 (74.94%)

dm.compile_dataset(filter_unmatched=True, num_proc=2, batch_size=10)
rich.print("[green]>> index is compiled.")

rich.print("=== Compiled Dataset ===")
rich.print(dm.compiled_dataset)

batch = next(iter(dm.train_dataloader()))
pprint_batch(batch, "compiled batch")

for idx in range(3):
    print(get_separator("-"))
    eg = Pipe.get_eg(batch, idx=idx)
    rich.print(
        f"Example #{idx}: \n"
        f" * answer=[magenta]{eg['answer.text'][eg['answer.target']]}[/magenta]\n"
        f" * question=[cyan]{eg['question.text']}[/cyan]\n"
        f" * documents: n_positive={sum(eg['document.is_positive'])}, "
        f"n_negative={sum(eg['document.is_positive'] == 0)}"
    )
    for j in range(min(len(eg["document.text"]), 3)):
        print(get_separator("."))
        match_on = eg.get("document.match_on", None)
        match_on = match_on[j] if match_on is not None else None
        rich.print(
            f" |-* document #{j}, score={eg['document.retrieval_score'][j]:.2f}, "
            f"is_positive={eg['document.is_positive'][j]}, match_on={match_on}"
        )
        txt = eg["document.text"][j].replace("\n", "")
        rich.print(f"[white]{txt}")

# dump data
for split, dset in dm.compiled_dataset.items():
    with open(f"compiled-dataset-{split}.jsonl", mode="w") as fp:
        for i, row in enumerate(dset):
            row["split"] = str(split)
            row["__index__"] = str(i)
            for key in list(row.keys()):
                if any(
                    ptrn in key for ptrn in ["input_ids", "attention_mask"]
                ):
                    row.pop(key)

            for k, v in row.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        row[k] = v.item()
                    else:
                        row[k] = str(v)

            fp.write(f"{json.dumps(row)}\n")
