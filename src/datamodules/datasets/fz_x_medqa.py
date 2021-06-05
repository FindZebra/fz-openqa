import json
from rich import print
import datasets


class FZxMedQAConfig(datasets.BuilderConfig):
    """BuilderConfig for FZxMedQA."""

    def __init__(self, **kwargs):
        """BuilderConfig for FZxMedQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FZxMedQAConfig, self).__init__(**kwargs)


_TRAIN_URL = (
    "https://drive.google.com/file/d/1P7qayfeZOE--W8Cblx3vuVDT6vyGb1LI/view?usp=sharing"
)
_VALID_URL = (
    "https://drive.google.com/file/d/1tcoj62M80c31PZCyugW13zMWpTSew9aj/view?usp=sharing"
)
_TEST_URL = (
    "https://drive.google.com/file/d/1T6idjeqwfifAf_MHIMWMw_bqeuT98OyT/view?usp=sharing"
)

_DESCRIPTION = "A mapping between the FinzdZebra corpus and the MedQA dataset"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""


class FZxMedQADataset(datasets.GeneratorBasedBuilder):
    """FZxMedQA Dataset. Version 0.0.1"""

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
                    "idx": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "question_id": datasets.Value("int32"),
                    "answer_idx": datasets.Value("int32"),
                    "answer_choices": datasets.Sequence(datasets.Value("string")),
                    "document": datasets.Value("string"),
                    "rank": datasets.Value("int32"),
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
                yield i, d
