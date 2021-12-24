from unittest import TestCase

from fz_openqa.datamodules.pipes.utils.spacy_pipe_functions import merge_consecutive_entities  # type: ignore
from fz_openqa.datamodules.pipes.relevance import AliasBasedMatch


class TestSpacyPipes(TestCase):
    def setUp(self) -> None:
        cls = AliasBasedMatch()
        self.model = cls._load_spacy_model(
            model_name="en_core_sci_lg",
            threshold=0.7,
            linker_name="umls"
        )

        self.exs = [
            "Increasing the heart rate decreases the relative amount of time spent during diastole",
            "The patient should receive both tetanus toxoid-containing vaccine and human tetanus immunoglobulin.",
            "Ketotifen eye drops",
            "Assess for suicidal ideation"
            ]

    def test_merge_consecutive_entities(self):
        output = list(self.model.pipe(self.exs))

        # {answer.text : "Increasing the heart rate decreases the relative amount of time spent during diastole"}. Expected output: (Increasing, heart rate decreases, time spent, diastole)  # noqa: E501
        self.assertEqual(len(output[0].ents), 4)
        # {answer.text : "The patient should receive both tetanus toxoid-containing vaccine and human tetanus immunoglobulin."}. Expected output: (patient, tetanus toxoid-containing vaccine, human tetanus) # noqa: E501
        self.assertEqual(len(output[1].ents), 3)
        # {answer.text : "Ketotifen eye drops"}. Expected output: (Ketotifen eye drops,) # noqa: E501
        self.assertEqual(len(output[2].ents), 1)
        # {answer.text : "Assess for suicidal ideation"}. Expected output: (Assess, suicidal ideation) # noqa: E501
        self.assertEqual(len(output[3].ents), 2)
