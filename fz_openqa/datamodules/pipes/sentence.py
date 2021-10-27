from fz_openqa.datamodules.pipes import Pipe


class GenerateSentences(Pipe):
    """A pipe to Extract sentences from a corpus of text.

    replication of: https://github.com/jind11/MedQA/blob/master/IR/scripts/insert_text_to_elasticsearch.py"""

    required_keys = ["text"]

    def __init__(
            self,
            *,
            delimitter: str,
            batch_size: int,
    ):
        self.delimitter = delimitter
        self.batch_size = batch_size

    def __call__(self, text: str) -> str:
        indexes, output = self.generate_passages_for_all_keys(
            batch,
            keys=["input_ids", "attention_mask", "offset_mapping"],
            args=self.args,
        )

        # extract document.text
        output["text"] = [
            self.extract_passage_text_from_doc(batch["text"][idx], ofs_ids)
            for idx, ofs_ids in zip(indexes, output["offset_mapping"])
        ]

        return output

    def groups(stream, size):
        batch = []
        for item in stream:
            batch += [item]
            if len(batch) % size == 0:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def text_to_sentences(self, text):
        for line in line_stream:
            line_cleaned = re.sub(r'([^a-zA-Z0-9\.])', " ", line).strip()
            for sentence in line_cleaned.split("."):
                if len(sentence) == 0:
                    continue
                yield sentence