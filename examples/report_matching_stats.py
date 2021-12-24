import argparse
import json

import datasets
import rich
import torch

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.corpus_dm import FZxMedQaCorpusDataModule
from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import ScispaCyMatch
from fz_openqa.datamodules.pipes import TextFormatter
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
parser.add_argument(
    "--filename",
    type=str,
    nargs="?",
    default="compiled-dataset",
    help="pre-fix to define filename",
)
args = parser.parse_args()

if args.cls == "scispacy":
    cls = ScispaCyMatch(interpretable=True, spacy_kwargs={"batch_size": 100, "n_process": 1})
elif args.cls == "metamap":
    cls = MetaMapMatch(interpretable=True)
elif args.cls == "exact":
    cls = ExactMatch(interpretable=True)
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

datasets.set_caching_enabled(True)
setup_safe_env()

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path="bert-base-cased")

# load the corpus object
corpus = corpus_module(
    tokenizer=tokenizer,
    index=ElasticSearchIndex(
        index_key="document.row_idx",
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
    train_batch_size=10,
    num_proc=4,
    num_workers=5,
    use_subset=args.subset,
    verbose=True,
    corpus=corpus,
    # retrieve 100 documents for each question
    n_retrieved_documents=args.topn,
    # keep only one positive doc
    max_pos_docs=10,
    # keep only 10 docs (1 pos + 9 neg)
    n_documents=100,
    # simple exact match
    relevance_classifier=cls,
    cache_dir="/scratch/s154097/cache",
)

# Compile the dataset
# ExactMatch: full dataset, num_proc=4, 1000 docs, bs=10: ~8s/batch, phoebe.compute.dtu.dk
# >  - train: 3473 (34.12%)
# >  - validation: 474 (37.26%)
# >  - test: 450 (35.35%)
# SciSpacyMatch: full dataset, num_proc=4, 1000 docs, bs=10: ~75s/batch, phoebe.compute.dtu.dk
# >  - train: 7605 (74.72%)
# >  - validation: 967 (76.02%)
# >  - test: 954 (74.94%)
# ExactMatch: FZ dataset, num_proc=4, 1000 docs, bs=10: ~4.6s/batch, phoebe.compute.dtu.dk
# >  - train: 2296 (22.56%)
# >  - validation: 338 (26.57%)
# >  - test: 306 (24.04%)
# ExactMatch: MedQA dataset, num_proc=4, 1000 docs, bs=10: ~3.6s/batch, phoebe.compute.dtu.dk
# >  - train: 2702 (26.55%)
# >  - validation: 354 (27.83%)
# >  - test: 371 (29.14%)
# ExactMatch: FZxMedQA dataset, num_proc=4, 1000 docs, bs=10: ~6s/batch, phoebe.compute.dtu.dk
# >  - train: 2487 (24.44%)
# >  - validation: 349 (27.44%)
# >  - test: 334 (26.24%)

# prepare both the QA dataset and the corpus
dm.subset_size = [1000, 100, 100]
dm.prepare_data()
dm.setup()

rich.print("=== Compiled Dataset ===")
rich.print(dm.dataset)

batch = next(iter(dm.train_dataloader()))
pprint_batch(batch, "compiled batch")

for idx in range(3):
    print(get_separator("-"))
    eg = Pipe.get_eg(batch, idx=idx)
    rich.print(
        f"Example #{1 + idx}: \n"
        f" * answer=[magenta]{eg['answer.text'][eg['answer.target']]}[/magenta]\n"
        f" * question=[cyan]{eg['question.text']}[/cyan]\n"
        f" * documents: n_positive={sum(eg['document.match_score'] > 0)}, "
        f"n_negative={sum(eg['document.match_score'] == 0)}"
    )
    for j in range(min(len(eg["document.text"]), 3)):
        print(get_separator("."))
        match_on = eg.get("document.match_on", None)
        match_on = match_on[j] if match_on is not None else None
        rich.print(
            f" |-* document #{1 + j}, score={eg['document.retrieval_score'][j]:.2f}, "
            f"match_score={eg['document.match_score'][j]}, match_on={match_on}"
        )
        txt = eg["document.text"][j].replace("\n", "")
        rich.print(f"[white]{txt}")


# dump data
for split, dset in dm.dataset.items():
    with open(f"{args.filename}-{split}.jsonl", mode="w") as fp:
        for i, row in enumerate(dset):
            row["split"] = str(split)
            row["__index__"] = str(i)
            for key in list(row.keys()):
                if any(ptrn in key for ptrn in ["input_ids", "attention_mask"]):
                    row.pop(key)

            for k, v in row.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        row[k] = v.item()
                    else:
                        row[k] = str(v)

            fp.write(f"{json.dumps(row)}\n")
