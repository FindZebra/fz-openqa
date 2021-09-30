import rich
import torch

from fz_openqa.utils.pretty import pprint_batch, get_separator


def gen_example_query(tokenizer):
    query = ["What is the symptoms of post polio syndrome?",
             "What are the symptoms of diabetes?"]
    batch_encoding = tokenizer(query)
    query = {**tokenizer.pad({k: [torch.tensor(vv) for vv in v]
                              for k, v in batch_encoding.items()}),
             'text': query}
    return {f"question.{k}": v for k, v in query.items()}


def display_search_results(query, results):
    pprint_batch(results)
    print(get_separator())
    for idx, (q, sub) in enumerate(
            zip(query['question.text'], results['document.text'])):
        print(get_separator("-"))
        rich.print(f"#{idx}: [magenta]{q}")
        for i, txt in enumerate(sub):
            rich.print(f"# index={i}")
            print(txt.strip())
