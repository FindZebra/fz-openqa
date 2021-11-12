from copy import copy

from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.utils.pretty import get_separator

formatter = TextFormatter(
    text_key="text",
    remove_breaks=True,
    remove_symbols=True,
    remove_ref=True,
)

text = [
    "(figure 7\xe2\x80\x9377) (figures 6\xe2\x80\x9328 and 6\xe2\x80\x9329) small nucleolar rna (snorna) ursor rrna. (table 6\xe2\x80\x931",  # noqa: E501
    "\t\t\t\t\t\t\t\tesophageal web stenosis in barium swallow examination lateral view.\t\t\t\t\t\t\t\t\t\t\t\t\tweb ",  # noqa: E501
    "some text \n nice shf aodfbks",
    "why\nisgfdks",
    "previously discussed methods (Smith, J. et al., 2014) ",
    ".\xe2\x80\x94\xe2\x80\x89passer and warnock",
    "people\xe2\x80\x94screens\xe2\x80\x94and that there were two screens",
    "differential\nPlatelet count: 25,000/mm^3\n\nSerum:\nNa+: 139",
    "\xe2\x80\x9cstepped down\xe2\x80\x9d to a lower",
]

batch = {"text": text}
new_batch = formatter(copy(batch))
for original_txt, new_txt in zip(batch["text"], new_batch["text"]):
    print(get_separator())
    print(f">> {original_txt}")
    print(f">> {new_txt}")
