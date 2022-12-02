import string
from typing import Optional

from pydantic import BaseModel
from warp_pipes import Eg

from fz_openqa.datamodules.utils.datastruct import Scenario


class LanguageModellingTemplate(BaseModel):
    question: str
    answer: Optional[str]

    @property
    def text(self):
        canvas = self.question
        if self.answer is not None:
            canvas = f"{canvas}{self.answer}"

        return canvas


class LanguageModellingTemplateGenerator:
    def __init__(
        self,
        scenario: Scenario,
        question_field: str = "question",
        answer_field: str = "answer",
        document_field: str = "document",
    ):
        self.scenario = scenario
        self.q_key = f"{question_field}.text"
        self.a_key = f"{answer_field}.text"
        self.d_key = f"{document_field}.text"
        self.a_target_key = f"{answer_field}.target"
        self.required_keys = [self.q_key, self.a_key]
        if scenario == Scenario.multiple_choice_qa:
            self.required_keys.append(self.a_target_key)

    def __call__(self, eg: Eg) -> LanguageModellingTemplate:
        input_keys = set(eg.keys())
        if not input_keys.issuperset(self.required_keys):
            raise ValueError(f"Requires keys {self.required_keys}. Found {eg.keys()}")

        canvas = ""
        if self.d_key in eg:
            doc = eg[self.d_key]
            if not isinstance(doc, str):
                raise ValueError(f"Document should be a string, found {type(doc)}")
            canvas += f"Context: {doc}\n\n"

        if self.scenario == Scenario.generative_qa:
            canvas += f"Q: {eg[self.q_key]}\nA: "
            answer_canvas = eg[self.a_key]
        elif self.scenario == Scenario.multiple_choice_qa:
            canvas += f"Q: {eg[self.q_key]}\n\n"
            letters = string.ascii_letters.upper()
            ans_repr = []
            for letter, opt in zip(letters, eg[self.a_key]):
                ans_repr.append(f"({letter}) {opt}")
            canvas += " ".join(ans_repr)

            target = eg[self.a_target_key]
            canvas += "\n\nA: The answer is ("
            # answer_canvas = f"{letters[target]}"
            answer_canvas = f"{letters[target]}) {eg[self.a_key][target]}"
        elif self.scenario == Scenario.multiple_choice_concat_qa:
            raise ValueError(
                "Template cannot be used for this scenario. "
                "You must run the template with scenario "
                "`Scenario.generative_qa` for each question-answer pair."
            )
        else:
            raise ValueError(f"Unknown scenario {self.scenario}")

        return LanguageModellingTemplate(question=canvas, answer=answer_canvas)
