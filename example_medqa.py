import numpy as np
import rich

from fz_openqa.datamodules.medqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path='bert-base-cased')

questions = MedQaDataModule(
                            tokenizer=tokenizer,
                            num_proc=4,
                            use_subset=True,
                            top_n_synonyms=3,
                            verbose=False)

questions.prepare_data()
questions.setup()

print(f">> Get questions")
rich.print(questions.dataset['train'])
rich.print(questions.dataset['train'][0])
