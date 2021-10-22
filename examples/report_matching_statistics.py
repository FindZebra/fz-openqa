import argparse
import json
from time import time

import datasets
import rich
import torch
from torch.utils.data import DataLoader

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.corpus_dm import FZxMedQaCorpusDataModule
from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import ScispaCyMatch
from fz_openqa.datamodules.pipes.relevance import MetaMapMatch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env

datasets.set_caching_enabled(True)
setup_safe_env()

parser = argparse.ArgumentParser(
    description="Generate reporting statistics on relevance classifiers"
)
parser.add_argument(
    "--cls",
    type=str,
    nargs="?",
    default="exact",
    help="classifier applied to label documents from the corpus; (exact, scispacy or metamap)",
)
parser.add_argument(
    "--corpus",
    type=str,
    nargs="?",
    default="FZ",
    help="corpus applied to generate documents from; (FZ, MedQA, FZxMedQA)",
)
parser.add_argument(
    "--topn",
    type=int,
    nargs="?",
    default=10,
    help="top n documents returned from ElasticSearch by a given query input",
)
parser.add_argument(
    "--filter",
    type=str,
    nargs="?",
    default=None,
    help="what filtering mode applied to the documents",
)
parser.add_argument(
    "--subset",
    type=bool,
    nargs="?",
    default=False,
    help="Run with a subset of corpus and question True or False",
)
args = parser.parse_args()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

if args.cls == "scispacy":
    cls = ScispaCyMatch()
elif args.cls == "metamap":
    cls = MetaMapMatch()
elif args.cls == "exact":
    cls = ExactMatch()
else:
    NotImplementedError

if args.corpus == "FZ":
    corpus_module = FzCorpusDataModule
elif args.corpus == "MedQA":
    corpus_module = MedQaCorpusDataModule
elif args.corpus == "FZxMedQA":
    corpus_module = FZxMedQaCorpusDataModule
else:
    NotImplementedError

# load the corpus object
corpus = corpus_module(
    tokenizer=tokenizer,
    index=ElasticSearchIndex(
        index_key="idx",
        text_key="document.text",
        query_key="question.text",
        num_proc=4,
        filter_mode=args.filter,
    ),
    verbose=False,
    num_proc=4,
    use_subset=args.subset,
    cache_dir="/scratch/s154097/cache",
)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    train_batch_size=100,
    num_proc=4,
    num_workers=4,
    use_subset=args.subset,
    verbose=True,
    corpus=corpus,
    # retrieve 100 documents for each question
    n_retrieved_documents=args.topn,
    # keep only one positive doc
    max_pos_docs=None,
    # keep only 10 docs (1 pos + 9 neg)
    n_documents=None,
    # simple exact match
    relevance_classifier=cls,
    compile_in_setup=False,
    cache_dir="/scratch/s154097/cache",
)

# prepare both the QA dataset and the corpus
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

# ExactMatch: FZ dataset, num_proc=4, 1000 docs, bs=10: ~8s/batch, phoebe.compute.dtu.dk
# >  - train: 2883 (28.33%)
# >  - validation: 424 (33.33%)
# >  - test: 374 (29.38%)

run_time_block = dm.compile_dataset(
    filter_unmatched=True, num_proc=4, batch_size=10
)

rich.print("[green]>> index is compiled.")

rich.print("=== Compiled Dataset ===")
rich.print(dm.compiled_dataset)

batch = next(iter(dm.train_dataloader()))
pprint_batch(batch, "compiled batch")

for k, x in run_time_block.items():
    rich.print(f"[red] Runtime for {k} : {x} seconds")

for idx in range(3):
    print(get_separator("-"))
    eg = Pipe.get_eg(batch, idx=idx)
    rich.print(
        f"Example #{idx}: \n"
        f" * answer=[magenta]{eg['answer.text'][eg['answer.target']]}[/magenta]\n"
        f" * question=[cyan]{eg['question.text']}[/cyan]\n"
        f" * documents: n_pos={sum(eg['document.is_positive'])}, "
        f"n_neg={sum(eg['document.is_positive'] == 0)}"
    )
    for j in range(min(len(eg["document.text"]), 3)):
        print(get_separator("."))
        rich.print(
            f" |-* doc #{j}, "
            f"score={eg['document.retrieval_score'][j]:.2f}, , "
            f"is_positive={eg['document.is_positive'][j]}"
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
