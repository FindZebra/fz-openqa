import json

import datasets


class MedQATwConfig(datasets.BuilderConfig):
    """BuilderConfig for the MedQa Taiwanese."""

    def __init__(self, **kwargs):
        """BuilderConfig for FZxMedQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedQATwConfig, self).__init__(**kwargs)


_TRAIN_URL = "https://drive.google.com/file/d/1RPQJEu2iRY-KPwgQBB2bhFWY-LJ-z9_G/view?usp=sharing"
_VALID_URL = "https://drive.google.com/file/d/1e-a6nE_HqnoQV_8k4YmaHbGSTTleM4Ag/view?usp=sharing"
_TEST_URL = "https://drive.google.com/file/d/13ISnB3mk4TXgqfu-JbsucyFjcAPnwwMG/view?usp=sharing"

_DESCRIPTION = "MedQA (tw) dataset - custom 4 options extraction"
_VERSION = "0.0.1"
_HOMEPAGE = ""
_CITATION = ""


class OfficialMedQaGenerator(datasets.GeneratorBasedBuilder):
    """MedQAxCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False
    urls = {
        datasets.Split.TRAIN: _TRAIN_URL,
        datasets.Split.VALIDATION: _VALID_URL,
        datasets.Split.TEST: _TEST_URL,
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question.idx": datasets.Value("int32"),
                    "question.text": datasets.Value("string"),
                    "question.metamap": datasets.Sequence(datasets.Value("string")),
                    "answer.target": datasets.Value("int32"),
                    "answer.text": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    @staticmethod
    def _get_drive_url(url):
        base_url = "https://drive.google.com/uc?id="
        split_url = url.split("/")
        return base_url + split_url[5]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = {
            split: dl_manager.download_and_extract(self._get_drive_url(url))
            for split, url in self.urls.items()
        }

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": file, "split": split},
            )
            for split, file in downloaded_files.items()
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, "r") as f:
            for i, d in enumerate(json.load(f)["data"]):
                # adjust values
                d["question.idx"] = d.pop("question_id")
                d["answer.target"] = d.pop("answer_idx")
                d["answer.text"] = d.pop("answer_options")
                d["question.text"] = d.pop("question")
                yield i, d
