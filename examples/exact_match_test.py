import rich

from fz_openqa.datamodules.pipes.relevance import ExactMatch, SciSpacyMatch

em = ExactMatch()
# todo mm = MetaMapMatch()
sm = SciSpacyMatch()

qst = "What is the symptoms of post polio syndrome?"
answer = {"answer.target": 0, "answer.text": ["Post polio syndrome (PPS)"]}
doc = {
    "document.text": "Post polio syndrome is a condition that affects polio survivors years after recovery from the initial polio illness. Symptoms and severity vary among affected people and may include muscle weakness and a gradual decrease in the size of muscles (atrophy); muscle and joint pain; fatigue;difficulty with gait; respiratory problems; and/or swallowing problems.\xa0Only a polio survivor can develop PPS. While polio is a contagious disease, PPS is not. The exact cause of PPS years after the first episode of polio is unclear, although several theories have been proposed. Treatment focuses on reducing symptoms and improving quality of life."}

rich.print(f"[cyan]Question: {qst}")
rich.print(f"[red]Answer: {answer['answer.text']}")

for classifier in [em, sm]:
    rich.print(f"> {type(classifier).__name__}: is_positive={classifier.classify(answer=answer, document=doc)}")
