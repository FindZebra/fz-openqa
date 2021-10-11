import datasets
import rich
from torch.utils.data import DataLoader

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch, Pipe
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env

datasets.set_caching_enabled(True)
setup_safe_env()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = MedQaCorpusDataModule(tokenizer=tokenizer,
                               index=ElasticSearchIndex(index_key="idx",
                                                        text_key="document.text",
                                                        query_key="question.text",
                                                        num_proc=4,
                                                        filter_mode=None),
                               verbose=False,
                               num_proc=4,
                               use_subset=False)

# load the QA dataset
dm = MedQaDataModule(tokenizer=tokenizer,
                     train_batch_size=100,
                     num_proc=4,
                     num_workers=1,
                     use_subset=True,
                     verbose=True,
                     corpus=corpus,
                     # retrieve 100 documents for each question
                     n_retrieved_documents=1000,
                     # keep only one positive doc
                     max_pos_docs=1,
                     # keep only 10 docs (1 pos + 9 neg)
                     n_documents=10,
                     # simple exact match
                     relevance_classifier=ExactMatch())

# prepare both the QA dataset and the corpus
dm.subset_size = [1000, 10, 10]
dm.prepare_data()
dm.setup()

print(get_separator())
dm.build_index()
rich.print(f"[green]>> index is built.")
print(get_separator())

# Compile the dataset
# ExactMatch: process speed: ~60s/batch
# SciSpacyMatch: process speed: ?/batch
dm.compile_dataset(filter_unmatched=True,
                   num_proc=4,
                   batch_size=10)
rich.print(f"[green]>> index is compiled.")

rich.print(f"=== Compiled Dataset ===")
rich.print(dm.compiled_dataset)


batch = next(iter(dm.train_dataloader()))
pprint_batch(batch, "compiled batch")

for idx in range(3):
    print(get_separator("-"))
    eg = Pipe.get_eg(batch, idx=idx)
    rich.print(f"Example #{idx}: \n"
               f" * answer=[magenta]{eg['answer.text'][eg['answer.target']]}[/magenta]\n"
               f" * question=[cyan]{eg['question.text']}[/cyan]\n"
               f" * documents: n_positive={sum(eg['document.is_positive'])}, n_negative={sum(eg['document.is_positive'] == 0)}")
    for j in range(min(len(eg['document.text']), 3)):
        print(get_separator("."))
        rich.print(
            f" |-* document #{j}, score={eg['document.retrieval_score'][j]:.2f}, , is_positive={eg['document.is_positive'][j]}")
        txt = eg['document.text'][j].replace("\n", "")
        rich.print(f"[white]{txt}")
