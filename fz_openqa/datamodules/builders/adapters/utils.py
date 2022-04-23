from typing import Any
from typing import Dict
from typing import List


def set_document_row_idx(
    row: Dict[str, Any],
    *,
    doc_uid_key: str,
    document_keys: List,
    output_key: str = "question.document_idx",
) -> Dict[str, Any]:
    row[output_key] = document_keys.index(row[doc_uid_key])
    return row
