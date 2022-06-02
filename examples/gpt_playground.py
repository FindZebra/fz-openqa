import abc
import math
import os
import string
from typing import Any
from typing import Dict
from typing import List

import openai
import rich
from dotenv import load_dotenv

from fz_openqa.utils.openai import AnsweringModel
from fz_openqa.utils.openai import MultipleChoiceTemplate

question = """\
A 27-year-old male presents to urgent care complaining of pain with urination. He reports that the pain started 3 days ago. He has never experienced these symptoms before. He denies gross hematuria or pelvic pain. He is sexually active with his girlfriend, and they consistently use condoms. When asked about recent travel, he admits to recently returning from a boys’ trip” in Cancun where he had unprotected sex 1 night with a girl he met at a bar. The patients medical history includes type I diabetes that is controlled with an insulin pump. His mother has rheumatoid arthritis. The patients temperature is 99 F (37.2 C), blood pressure is 112/74 mmHg, and pulse is 81/min. On physical examination, there are no lesions of the penis or other body rashes. No costovertebral tenderness is appreciated. A urinalysis reveals no blood, glucose, ketones, or proteins but is positive for leukocyte esterase. A urine microscopic evaluation shows a moderate number of white blood cells but no casts or crystals. A urine culture is negative. Which of the following is the most likely cause for the patient’s symptoms?\
"""  # noqa
options = [
    "Chlamydia trachomatis",
    "Systemic lupus erythematosus",
    "Mycobacterium tuberculosis",
    "Treponema pallidum",
]


engine = "text-ada-001"
# engine = "text-curie-001"
engine = "text-davinci-002"


model = AnsweringModel(
    engine=engine,
    prompt_mode="chain_of_thought",
    template=MultipleChoiceTemplate(),
)

answer, diagnostics = model(question, options)
rich.print(f"[magenta]The answer is {answer}")
for k, v in diagnostics.items():
    k_ = f" {k} "
    rich.print(f"[blue]{k_:=^60}")
    rich.print(v)
