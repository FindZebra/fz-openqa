import json

import datasets

class WikiCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for the MedQa English Corpus objecxt."""

    def __init__(self, **kwargs):
        """BuilderConfig for the Corpus object.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiCorpusConfig, self).__init__(**kwargs)


_DESCRIPTION = "A class to load a collection of MedQA related Wikipedia articles"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""
_URL = "https://drive.google.com/file/d/1DtLbMsErbFOQ7u-kDz_tgb8AR_YiRTUz/view?usp=sharing"


class WikiCorpusGenerator(datasets.GeneratorBasedBuilder):
    """FzCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "title": datasets.Value("string"),
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
        downloaded_file = dl_manager.download_and_extract(self._get_drive_url(_URL))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_file},
            )
        ]

    def _generate_examples(self, filepath: str):
        """Yields examples."""

        with open(filepath, "r") as json_file:
            json_list = list(json_file)
            for idx, val in enumerate(json_list):
                article = json.loads(val)
                yield idx, {
                    "text": article["content"],
                    "title": article["title"],
                    "idx": idx,
                }
