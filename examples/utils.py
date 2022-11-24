from typing import Dict

import rich
import torch
from warp_pipes import pprint_batch
from warp_pipes.support.pretty import get_console_separator


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
    print(get_console_separator())
    for idx, (qst, row_idxs, scores, tokens) in enumerate(
        zip(
            queries["question.text"],
            results["document.row_idx"],
            results["document.proposal_score"],
            results["document.analyzed_tokens"],
        )
    ):
        print(get_console_separator("-"))
        rich.print(f"#{idx}: [magenta]{qst}")
        for i, row_idx in enumerate(row_idxs):
            rich.print(f"# index={i}")
            print(corpus[row_idx]["document.text"].strip())
            print(tokens[i])
            print(scores[i])
