import math

import openai
import rich

openai.api_key = ...

engine = "text-ada-001"
# engine = "text-davinci-002"
# engine = "text-curie-001"


def extract_answer(answer_text, options):
    def safe_index(txt, value, default_index):
        if value in txt:
            return txt.index(value)
        else:
            return default_index

    indices = [(o, safe_index(answer_text, o, None)) for o in options]
    indices = list(filter(lambda x: x[1] is not None, indices))
    if len(indices):
        return min(indices, key=lambda x: x[1])[0]
    else:
        return None


prompt = """
Q: A 27-year-old male presents to urgent care complaining of pain with urination. He reports that the pain started 3 days ago. He has never experienced these symptoms before. He denies gross hematuria or pelvic pain. He is sexually active with his girlfriend, and they consistently use condoms. When asked about recent travel, he admits to recently returning from a boys’ trip” in Cancun where he had unprotected sex 1 night with a girl he met at a bar. The patients medical history includes type I diabetes that is controlled with an insulin pump. His mother has rheumatoid arthritis. The patients temperature is 99 F (37.2 C), blood pressure is 112/74 mmHg, and pulse is 81/min. On physical examination, there are no lesions of the penis or other body rashes. No costovertebral tenderness is appreciated. A urinalysis reveals no blood, glucose, ketones, or proteins but is positive for leukocyte esterase. A urine microscopic evaluation shows a moderate number of white blood cells but no casts or crystals. A urine culture is negative. Which of the following is the most likely cause for the patient’s symptoms?
Answer options:
A) Chlamydia trachomatis
B) Systemic lupus erythematosus
C) Mycobacterium tuberculosis
D) Treponema pallidum

A: Let's think step by step like a medical expert.
"""  # noqa

response = openai.Completion.create(
    engine=engine,
    prompt=prompt,
    # suffix="the answer is  option (A) Chlamydia trachomatis.",
    temperature=0,
    max_tokens=300,
    top_p=1,
    logprobs=5,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["<|endoftext|>"],
)

j = 0
rich.print("# step 1: reasoning")
answer = response["choices"][0]["text"]
log_probs = response["choices"][0]["logprobs"]
# rich.print(f"> Answer: {answer}")

tkens = log_probs["tokens"]
tkens_logprobs = log_probs["token_logprobs"]
rich.print(f"> {''.join(tkens)}: {sum(tkens_logprobs)}")

rich.print("# step 2: extraction")
prompt += answer.split("<|endoftext|>")[0] + " Therefore, among A through D, the answer is "

response = openai.Completion.create(
    engine=engine,
    prompt=prompt,
    # suffix="the answer is  option (A) Chlamydia trachomatis.",
    temperature=0,
    max_tokens=300,
    top_p=1,
    logprobs=5,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["."],
)

answer = response["choices"][0]["text"]
log_probs = response["choices"][0]["logprobs"]
# rich.print(f"> Answer: {answer}")

tkens = log_probs["tokens"]
tkens_logprobs = log_probs["token_logprobs"]
rich.print(f"> {''.join(tkens)}: {sum(tkens_logprobs)}")
prompt += answer.split("<|endoftext|>")[0]


rich.print("[green]=== full output ===")
rich.print(prompt)

rich.print(f"[magenta] Extracted answer: {extract_answer(answer, options=['A', 'B', 'C', 'D'])}")
