import logging
import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import datasets
import hydra
import rich
from hydra.utils import instantiate
from omegaconf import DictConfig

import fz_openqa
from fz_openqa import configs
from fz_openqa.datamodules.analytics.corpus_statistics import ReportCorpusStatistics
from fz_openqa.datamodules.builders.corpus import FzCorpusBuilder
from fz_openqa.datamodules.builders.fz_queries import FzQueriesBuilder
from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.index_pipes import FetchDocuments
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)

DEFAULT_CKPT = "https://drive.google.com/file/d/11IgxWOVFLcSqiIwtWXgDEQ1NpNdl-egL/view?usp=sharing"
default_cache_dir = Path(fz_openqa.__file__).parent.parent / "cache"


class ComputeCuiRank(Pipe):
    def __init__(
        self,
        question_cui_key: str = "question.cui",
        document_cui_key: str = "document.cui",
        document_score_key: str = "document.retrieval_score",
        output_key: str = "question.document_rank",
        output_match_id_key: str = "question.matched_document_idx",
        method: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question_cui_key = question_cui_key
        self.document_cui_key = document_cui_key
        self.document_score_key = document_score_key
        self.output_match_id_key = output_match_id_key
        self.method = method
        self.output_key = output_key

    @staticmethod
    def _lower(x: Any) -> Any:
        if isinstance(x, str):
            return x.lower()
        elif isinstance(x, list):
            return [ComputeCuiRank._lower(y) for y in x]
        else:
            raise ValueError(f"Unsupported type {type(x)}")

    @staticmethod
    def _rank(
        q_cuis: List[str], d_cuis: List[str], d_scores: List[float], method: str = "sum"
    ) -> Tuple[int, List[int]]:
        rank_data = [
            ComputeCuiRank._rank_one_q_cui(q_cui, d_cuis, d_scores, method=method)
            for q_cui in q_cuis
        ]
        rank_data = [(r, ids) for r, ids in rank_data if r is not None]
        if len(rank_data) == 0:
            return -1, []
        else:
            ranks, ids = min(rank_data, key=lambda x: x[0])
            return ranks, ids

    @staticmethod
    def _rank_one_q_cui(
        q_cui: str, d_cuis: List[str], d_scores: List[float], method: str = "sum"
    ) -> Tuple[Optional[int], List[int]]:
        """Compute the rank of a query cui in a list of document cuis and
        return the rank along the index of the matched documents."""
        if q_cui not in d_cuis:
            return None, []
        init_value = 0.0 if method == "sum" else -float("inf")
        cui_scores = defaultdict(lambda: init_value)
        cui_ids = defaultdict(list)
        for i, (d_cui, d_score) in enumerate(zip(d_cuis, d_scores)):
            cui_ids[d_cui] += [i]
            if method == "sum":
                if d_score < -1e3:
                    logger.warning(f"Score {d_score} is low.")
                cui_scores[d_cui] += d_score
            else:
                cui_scores[d_cui] = max(cui_scores[d_cui], d_score)

        # sort document CUIs by score
        ranked_cuis = sorted(cui_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_cuis = [x[0] for x in ranked_cuis]

        # return the rank of the question cui
        return ranked_cuis.index(q_cui), cui_ids.get(q_cui, [])

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        """Compute the rank of the query CUIs in the retrieved documents and
        return the ids of the corresponding documents.

        Notes
        -----
        Documents ids correspond to the index of the document in the batch, not the index
        of the document in the dataset."""
        q_batch_cuis = self._lower(batch[self.question_cui_key])
        d_batch_cuis = self._lower(batch[self.document_cui_key])
        d_batch_scores = batch[self.document_score_key]

        # compute ranks and fetch the corresponding document ids
        document_ranks = []
        document_match_ids = []
        for q_cuis, d_cuis, d_scores in zip_longest(
            q_batch_cuis,
            d_batch_cuis,
            d_batch_scores,
        ):
            # get the rank of the question CUIs in the retrieved documents
            rank, matched_ids = self._rank(q_cuis, d_cuis, d_scores, method=self.method)
            document_ranks += [rank]
            document_match_ids += [matched_ids]

        # format output and return
        output = {self.output_key: document_ranks, self.output_match_id_key: document_match_ids}

        return output


class FetchCuiAndRank(Sequential):
    def __init__(
        self,
        corpus_dataset: datasets.Dataset,
        *,
        question_cui_key: str = "question.cui",
        document_cui_key: str = "document.cui",
        document_score_key: str = "document.retrieval_score",
        document_id_key: str = "document.row_idx",
        output_key: str = "question.document_rank",
        output_match_id_key: str = "question.matched_document_idx",
        method: str = "sum",
        level: int = 1,
        **kwargs,
    ):
        super().__init__(
            ApplyAsFlatten(
                FetchDocuments(corpus_dataset=corpus_dataset, keys=[document_cui_key]),
                input_filter=In([document_id_key]),
                update=True,
                level=level,
            ),
            ComputeCuiRank(
                question_cui_key=question_cui_key,
                document_cui_key=document_cui_key,
                document_score_key=document_score_key,
                output_key=output_key,
                output_match_id_key=output_match_id_key,
                method=method,
            ),
            **kwargs,
        )


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    cache_dir = config.get("sys.cache_dir", default_cache_dir)

    # load a checkpoint
    loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
    # loader.print_config()

    # initialize the corpus builder
    corpus_builder: FzCorpusBuilder = loader.instantiate(
        "datamodule.builder.corpus_builder",
        _target_=FzCorpusBuilder,
        to_sentences=config.get("to_sentences", False),
        use_subset=config.get("corpus_subset", False),
        cache_dir=cache_dir,
        num_proc=config.get("num_proc", 4),
        analyses=[ReportCorpusStatistics(output_dir="./analyses", verbose=True)],
    )

    # initialize the queries builder
    dataset_builder: FzQueriesBuilder = loader.instantiate(
        "datamodule.builder.dataset_builder",
        _target_=FzQueriesBuilder,
        use_subset=config.get("dataset_subset", False),
        cache_dir=cache_dir,
        num_proc=config.get("num_proc", 4),
    )

    # build
    corpus = corpus_builder()
    rich.print(f"=== corpus : fingerprint={corpus._fingerprint} ===")
    rich.print(corpus)
    dataset = dataset_builder()["test"]
    rich.print(f"=== dataset : fingerprint={dataset._fingerprint} ===")
    rich.print(dataset)

    # load model
    model = None  # loader.load_model()

    # define the trainer
    trainer = instantiate(config.get("trainer", {}))

    # build the index
    index_builder = instantiate(
        config.datamodule.index_builder,
        # _target_=FaissIndexBuilder,
        model=model,
        trainer=trainer,
        # text_formatter=TextFormatter(aggressive_cleaning=True), # todo: remove
        map_kwargs={"num_proc": 4, "batch_size": 100},
        loader_kwargs={"num_workers": 4, "batch_size": 10},
        # window_size=200, # todo
        # window_stride=200, # todo
        dataset=corpus,
        persist_cache=True,
        cache_dir=cache_dir,
    )

    logger.info("Indexing corpus..")
    index: Index = index_builder()

    # query the index with the dataset
    logger.info(f"Indexing dataset (pipe={index.fingerprint(reduce=True)})..")
    indexed_dataset = index(
        dataset,
        k=config.get("topk", 1000),
        batch_size=10,
        trainer=trainer,
        num_proc=4,
        cache_fingerprint=Path(cache_dir) / "rank_fz_queries_fingerprints" / "index",
        fingerprint_kwargs_exclude=["trainer"],
        deterministic_fingerprint=True,
    )
    pprint_batch(indexed_dataset[:10], "search output")

    # compute the rank
    logger.info("Ranking dataset..")
    ranker = FetchCuiAndRank(corpus, method="sum")
    indexed_dataset = ranker(
        indexed_dataset,
        batch_size=10,
        num_proc=4,
        cache_fingerprint=Path(cache_dir) / "rank_fz_queries_fingerprints" / "ranker",
        deterministic_fingerprint=True,
    )
    pprint_batch(indexed_dataset[:10], "ranker output")

    # display questions and retrieved passages
    indices = slice(20, 23)
    fetch_doc_data = FetchDocuments(
        corpus_dataset=corpus, index_key="document.row_idx", keys=["document.text", "document.cui"]
    )
    batch = indexed_dataset[indices]
    for i in range(len(batch["question.text"])):
        print(get_separator("-"))
        q_txt = batch["question.text"][i]
        q_txt = q_txt.replace("'", "")
        rich.print(f"#{i + 1} - cui={batch['question.cui'][i]}, " f"query=`[white]{q_txt}[/white]`")

        # retrieve the index of the matched documents within the corpus
        # given the local indexes
        matched_local_ids = batch["question.matched_document_idx"][i]
        batch_row_ids = batch["document.row_idx"][i]
        matched_row_ids = [batch_row_ids[i] for i in matched_local_ids]

        # fetch the document data given the document row_idx
        docs = fetch_doc_data({"document.row_idx": matched_row_ids})

        # display the matched documents
        for j in range(min(3, len(matched_local_ids))):
            rich.print(get_separator("."))
            loc_j = matched_local_ids[j]
            doc_txt = docs["document.text"][j]
            doc_txt = doc_txt.replace("'", "")
            rich.print(
                f"- doc #{j+1}, "
                f"score={batch['document.retrieval_score'][i][loc_j]:.2f}, "
                f"cui={docs['document.cui'][j]}, "
                f"text=`[white]{doc_txt}[/white]`"
            )

    # compute recall
    ranks = indexed_dataset["question.document_rank"]
    for target_rank in [1, 5, 20, 50, 100, 1000]:
        recall = ((ranks >= 0).float() * (ranks < target_rank).float()).sum() / len(ranks)
        rich.print(f"Recall@{target_rank}={recall * 100:.2f}%")


if __name__ == "__main__":
    run()
