from typing import Optional

from datasets import Dataset
from datasets import DatasetDict
from warp_pipes import keep_only_columns

from fz_openqa.datamodules.builders.adapters.base import DatasetAdapter
from fz_openqa.datamodules.pipes.text_formatter import ReSubPatternFormatter

COLUMNS = ["text", "persistent_id", "source_url", "title", "cui", "source"]


class FindZebraCorpusAdapter(DatasetAdapter):
    """Adapt the FindZebra corpus (e.g., `findzebra/corpus.latest`) to the fz-openqa format
    and clean the text."""

    def __call__(
        self, dataset: DatasetDict, **kwargs
    ) -> (Optional[DatasetDict], Optional[Dataset]):
        dataset = dataset.rename_column("raw_content", "text")
        dataset = keep_only_columns(dataset, COLUMNS)

        # format the text, the simpler `ReSubPatternFormatter` happened to
        # yield better results than `HtmlCleaner` in the retrieval task,
        # even if the latter is arguably more powerful..
        formatter = ReSubPatternFormatter(text_key="text", clean_pattern=r"(<.*?>)|(\[.*?\])")
        # formatter = HtmlCleaner(text_key="text")

        dataset = formatter(dataset, desc="Clean corpus content (html)", **kwargs)
        return None, dataset["train"]
