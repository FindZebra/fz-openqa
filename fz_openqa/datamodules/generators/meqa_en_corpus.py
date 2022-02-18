import os
import re
from pathlib import Path

import datasets

TXT_PATTERN = r"^.*\.txt$"


class MedQaEnCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for the MedQa English Corpus objecxt."""

    def __init__(self, **kwargs):
        """BuilderConfig for the Corpus object.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedQaEnCorpusConfig, self).__init__(**kwargs)


_DESCRIPTION = "A class to load the english MedQA corpus"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""
_URL = "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/medqa-corpus-en.zip"


class MedQaEnCorpusGenerator(datasets.GeneratorBasedBuilder):
    """MedQaEnCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "document.idx": datasets.Value("int32"),
                    "document.text": datasets.Value("string"),
                    "document.title": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    @staticmethod
    def _get_drive_url(url):
        if "drive.google.com" in url:
            base_url = "https://drive.google.com/uc?id="
            split_url = url.split("/")
            return base_url + split_url[5]
        else:
            return url

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(self._get_drive_url(_URL))
        if not Path(downloaded_file).is_dir():
            raise Exception(
                f"Could not download the dataset Content of `downloaded_file`:"
                f"{open(downloaded_file, 'r').read()}"
            )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"output_dir": Path(downloaded_file) / "en"},
            )
        ]

    def _generate_examples(self, output_dir: str):
        """Yields examples."""
        data_files = [
            os.path.join(output_dir, p)
            for p in os.listdir(output_dir)
            if re.findall(TXT_PATTERN, p)
        ]
        for i, fn in enumerate(data_files):
            with open(fn, "r") as f:
                yield i, {"document.text": f.read(), "document.idx": i, "document.title": ""}
