from time import time

import datasets
import rich
from rich.progress import track

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes.relevance import ExactMatch, MetaMapMatch, SciSpacyMatch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch

em = ExactMatch()
mm = MetaMapMatch()
sm = SciSpacyMatch()

qst = "What is the symptoms of post polio syndrome?"
answer = {"answer.target":0, "answer.text": ["Post polio syndrome (PPS)"]}
doc = {"document.text": "Post polio syndrome is a condition that affects polio survivors years after recovery from the initial polio illness. Symptoms and severity vary among affected people and may include muscle weakness and a gradual decrease in the size of muscles (atrophy); muscle and joint pain; fatigue;difficulty with gait; respiratory problems; and/or swallowing problems.\xa0Only a polio survivor can develop PPS. While polio is a contagious disease, PPS is not. The exact cause of PPS years after the first episode of polio is unclear, although several theories have been proposed. Treatment focuses on reducing symptoms and improving quality of life."}

rich.print(f"[cyan]Question: {qst}")
rich.print(f"[red]Answer: {answer['answer.text']}")

rich.print("ExactMatch positive label:",em.classify(answer=answer, document=doc))
#rich.print("MetaMapMatch positive label:",mm.classify(answer=answer, document=doc))
rich.print("SciSpacy positive label:",sm.classify(answer=answer, document=doc))