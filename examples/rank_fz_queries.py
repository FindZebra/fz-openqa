import logging
import os
import sys
from pathlib import Path

import datasets
import hydra
import rich
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.analytics.corpus_statistics import ReportCorpusStatistics
from fz_openqa.datamodules.builders.corpus import FzCorpusBuilder
from fz_openqa.datamodules.builders.fz_queries import FzQueriesBuilder
from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.index_pipes import FetchDocuments
from fz_openqa.datamodules.index.rank import FetchCuiAndRank
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.modeling import Model
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.pretty import get_separator

logger = logging.getLogger(__name__)

default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="rank_config.yaml",
)
def run(config: DictConfig) -> None:
    """"
    Rank the FindZebra queries using Indexes.


    Notes
    -----
    1. Running with ElasicSearch and ScispaCy filtering:
    ```bash
    poetry run python examples/rank_fz_queries.py datamodule/index_builder/text_filter=scispacy_lg
    ```

    2. Running with the pretrained model (dense index):
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python examples/rank_fz_queries.py \
      +environ=titan trainer.strategy=dp trainer.gpus=8 +batch_size=1000  \
      datamodule/index_builder=dense


    Results:
    --------
        ElasticSearch:
         - Recall@    1: 2.02%
         - Recall@    5: 37.50%
         - Recall@   20: 57.66%
         - Recall@   50: 71.77%
         - Recall@  100: 76.61%
         - Recall@ 1000: 87.90%

        Dense (CLS):
         - Recall@    1: 0.00%
         - Recall@    5: 3.63%
         - Recall@   20: 8.47%
         - Recall@   50: 15.73%
         - Recall@  100: 22.58%
         - Recall@ 1000: 46.37%

         Dense (Colbert):
          - Recall@    1: 0.40%
          - Recall@    5: 3.63%
          - Recall@   20: 10.48%
          - Recall@   50: 17.34%
          - Recall@  100: 20.97%
          - Recall@ 1000: 36.29%



    ```
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    # avoid "too many open files" error
    torch.multiprocessing.set_sharing_strategy("file_system")

    # set the default checkpoint based on the environment
    if os.environ.get("USER") == "valv" and sys.platform != "darwin":
        if any("colbert" in arg for arg in sys.argv):
            DEFAULT_CKPT = "/scratch/valv/fz-openqa/runs/2021-12-16/17-36-40"
        else:
            DEFAULT_CKPT = "/scratch/valv/fz-openqa/runs/2021-12-17/10-39-57"
    else:
        DEFAULT_CKPT = (
            "https://drive.google.com/file/d/11IgxWOVFLcSqiIwtWXgDEQ1NpNdl-egL/view?usp=sharing"
        )

    # load a checkpoint
    # todo: cleanup config overriding
    # todo: don't use hydra main, catch command line overriding from `CheckpointLoader`
    #  https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
    loader.print_config()
    cache_dir = Path(loader.config.sys.cache_dir)
    rich.print(f">> cache_dir = {cache_dir}, exists={cache_dir.exists()}")

    # load model
    model: Model = loader.load_model(zero_shot=config.get("zero_shot", False))
    logger.info(f"Loaded model {type(model.module)}, fingerprint={get_fingerprint(model)}")

    # initialize the corpus builder
    logging.info(
        f"Initializing corpus builder "
        f"<{loader.config.datamodule.builder.corpus_builder._target_}>"
    )
    corpus_builder: FzCorpusBuilder = loader.instantiate(
        "datamodule.builder.corpus_builder",
        analyses=[ReportCorpusStatistics(output_dir="./analyses", verbose=True)],
    )

    # initialize the queries builder
    logging.info(
        f"Initializing dataset builder "
        f"<{loader.config.datamodule.builder.dataset_builder._target_}>"
    )
    dataset_builder: FzQueriesBuilder = loader.instantiate(
        "datamodule.builder.dataset_builder",
    )

    # build
    corpus = corpus_builder()
    rich.print(f"=== corpus : fingerprint={corpus._fingerprint} ===")
    rich.print(corpus)
    dataset = dataset_builder()["test"]
    rich.print(f"=== dataset : fingerprint={dataset._fingerprint} ===")
    rich.print(dataset)

    # define the trainer
    trainer = instantiate(config.get("trainer", {}))
    rich.print(f">> trainer = {trainer}")

    # build the index
    index_builder = instantiate(
        config.datamodule.index_builder,
        model=model,
        trainer=trainer,
        dataset=corpus,
        collate_pipe=corpus_builder.get_collate_pipe(
            columns=["document.text", "document.input_ids", "document.attention_mask"]
        ),
    )

    logger.info(f"Indexing index with {index_builder}..")
    index: Index = index_builder()

    # query the index with the dataset
    logger.info(f"Indexing dataset (pipe={index.fingerprint(reduce=True)})..")
    indexed_dataset = index(
        dataset,
        collate_fn=dataset_builder.get_collate_pipe(
            columns=["question.input_ids", "question.attention_mask"]
        ),
        k=config.get("topk", 1000),
        batch_size=10,
        trainer=trainer,
        num_proc=4,
        cache_fingerprint=Path(cache_dir) / "rank_fz_queries_fingerprints",
        fingerprint_kwargs_exclude=["trainer"],
        set_new_fingerprint=True,
    )

    # compute the rank
    logger.info("Ranking dataset..")
    ranker = FetchCuiAndRank(corpus, method="sum", fetch_max_chunk_size=100)
    indexed_dataset = ranker(
        indexed_dataset,
        batch_size=10,
        # writer_batch_size=3000,
        num_proc=4,
        cache_fingerprint=Path(cache_dir) / "rank_fz_queries_fingerprints",
        set_new_fingerprint=True,
    )

    # display questions and retrieved passages
    indices = slice(40, 50)
    fetch_doc_data = FetchDocuments(
        corpus_dataset=corpus,
        index_key="document.row_idx",
        keys=["document.text", "document.cui", "document.title"],
    )
    batch = indexed_dataset[indices]
    for i in range(len(batch["question.text"])):
        print(get_separator("-"))
        q_txt = batch["question.text"][i]
        q_txt = q_txt.replace("'", "")
        rich.print(
            f"#{i + 1} - cui={batch['question.cui'][i]}, "
            f"match_rank={batch['question.document_rank'][i]}, "
            f"query=`[white]{q_txt}[/white]`"
        )

        # get the local ids (position in batch)
        matched_local_ids = batch["question.matched_document_idx"][i]
        matched_local_ids = list(filter(lambda x: x >= 0, matched_local_ids))

        if len(matched_local_ids) == 0:
            rich.print("- [red]No matched documents[/red]")
            continue

        # get the global ids (row_idx)
        batch_row_ids = batch["document.row_idx"][i]
        matched_row_ids = [batch_row_ids[i] for i in matched_local_ids]

        # fetch the document data given the document row_idx
        docs = fetch_doc_data({"document.row_idx": matched_row_ids})

        # display the matched documents
        for j in range(min(3, len(matched_local_ids))):
            rich.print(get_separator("."))
            loc_rank = matched_local_ids[j]
            doc_txt = docs["document.text"][j]
            doc_txt = doc_txt.replace("'", "")
            rich.print(
                f"- doc #{j + 1}, local_rank={loc_rank + 1}, "
                f"score={batch['document.retrieval_score'][i][loc_rank]:.2f}, "
                f"cui={docs['document.cui'][j]}, "
                f"text=`[white]{doc_txt}[/white]`"
            )

    # compute recall
    print(get_separator())
    ranks = indexed_dataset["question.document_rank"]
    for target_rank in [1, 5, 20, 50, 100, 1000]:
        recall = ((ranks >= 0).float() * (ranks < target_rank).float()).sum() / len(ranks)
        rich.print(f" - Recall@{target_rank:5}: {recall * 100:.2f}%")

    # save all top-k retrieved passages
    batch_size = 100
    topk = 10
    output_path = Path(config.sys.work_dir) / "outputs"
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = (
        output_path
        / f'ranking-{", ".join(sys.argv[1:]).replace(" ","").replace("/",".").replace("+","")}.txt'
    )
    logger.info(f"Saving top-{topk} retrieved docs to `{output_path.absolute()}`")
    with open(str(output_path), "w") as f:
        for i in range(0, len(indexed_dataset), batch_size):
            batch = indexed_dataset[i : i + batch_size]
            for i in range(len(batch["question.text"])):
                q_txt = batch["question.text"][i]
                q_txt = q_txt.replace("'", "")
                f.write(100 * "=" + "\n")
                f.write(
                    f" - query #{i + 1} - cui={batch['question.cui'][i]}, "
                    f"match_rank={batch['question.document_rank'][i]}, "
                    f"query=`{q_txt}`\n"
                )

                # get the global ids (row_idx)
                batch_row_ids = batch["document.row_idx"][i][:topk]

                # fetch the document data given the document row_idx
                docs = fetch_doc_data({"document.row_idx": batch_row_ids})

                # display the matched documents
                for j in range(len(docs["document.text"])):
                    f.write(100 * "." + "\n")
                    doc_txt = docs["document.text"][j]
                    doc_txt = doc_txt.replace("'", "").replace("\n", " ")
                    q_cuis = batch["question.cui"][i]
                    is_match = any(
                        q_cui.lower() == docs["document.cui"][j].lower() for q_cui in q_cuis
                    )
                    f.write(
                        f"- doc #{j + 1}, query #{i+1}, "
                        f"score={batch['document.retrieval_score'][i][j]:.2f}, "
                        f"title={docs['document.title'][j]}, "
                        f"match={is_match}, "
                        f"text=`{doc_txt}`\n"
                    )


if __name__ == "__main__":
    run()
