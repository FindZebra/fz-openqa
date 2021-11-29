from typing import Any
from typing import Dict

import pytorch_lightning as pl
import rich
import torch
from transformers import AutoModel

from fz_openqa.datamodules.index.utils.es_engine import ElasticSearchEngine
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch

es_engine = ElasticSearchEngine()


def gen_example_query(tokenizer):
    query = [
        "What is the symptoms of post polio syndrome?",
        "What are the symptoms of diabetes?",
    ]
    batch_encoding = tokenizer(query)
    query = {
        **tokenizer.pad({k: [torch.tensor(vv) for vv in v] for k, v in batch_encoding.items()}),
        "text": query,
    }
    return {f"question.{k}": v for k, v in query.items()}


def display_search_results(corpus, queries: Dict, results: Dict):
    pprint_batch(results)
    print(get_separator())
    for idx, (qst, row_idxs, scores, tokens) in enumerate(
        zip(
            queries["question.text"],
            results["document.row_idx"],
            results["document.retrieval_score"],
            results["document.analyzed_tokens"],
        )
    ):
        print(get_separator("-"))
        rich.print(f"#{idx}: [magenta]{qst}")
        for i, row_idx in enumerate(row_idxs):
            rich.print(f"# index={i}")
            print(corpus[row_idx]["document.text"].strip())
            print(tokens[i])
            print(scores[i])


class ZeroShot(pl.LightningModule):
    def __init__(
        self, bert_id: str = "dmis-lab/biobert-base-cased-v1.2", head: str = "flat", **kwargs
    ):
        super(ZeroShot, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_id)
        assert head in {"flat", "contextual"}
        self.head = head

    def forward(self, batch: Batch, **kwargs) -> Any:
        output = {}
        key_map = {"document": "_hd_", "question": "_hq_"}
        for prefix in ["document", "question"]:
            if any(prefix in k for k in batch.keys()):
                input_ids = batch[f"{prefix}.input_ids"]
                attention_mask = batch[f"{prefix}.attention_mask"]
                h = self.bert(input_ids, attention_mask).last_hidden_state

                if self.head == "flat":
                    vec = h[:, 0, :]
                elif self.head == "contextual":
                    vec = h / h.norm(dim=2, keepdim=True)
                else:
                    raise NotImplementedError

                output[key_map[prefix]] = vec

        return output
