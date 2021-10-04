from unittest import TestCase

from fz_openqa.datamodules.pipes.relevance import MetaMapMatch, SciSpacyMatch, ExactMatch

class TestClassifier(TestCase):
    def setUp(self) -> None:
        self.answer = {'answer.target': 0, 'answer.text': "hej"}
        self.document = {'document.text': "once upon a time"}
        self.model_name = "en_core_sci_lg"

    def test_exact_match(self):
        self.assertTrue(ExactMatch.classify(answer = self.answer, document = self.document))

    def test_metamap_match(self):
        self.assertTrue(MetaMapMatch.classify(answer = self.answer, document = self.document, model_name=self.model_name))

    def test_scispacy_match(self):
        self.assertTrue(SciSpacyMatch.classify(answer = self.answer, document = self.document, model_name=self.model_name))
