from copy import copy

import rich

from fz_openqa.datamodules.pipes import SciSpacyFilter
from fz_openqa.utils.pretty import get_separator

filter = SciSpacyFilter(text_key="text")
txt = [
    "A 59-year-old overweight woman presents to the urgent care clinic with "
    "the complaint of severe abdominal pain for the past 2 hours. She also "
    "complains of a dull pain in her back with nausea and vomiting several times. "
    "Her pain has no relation with food. Her past medical history is significant "
    "for recurrent abdominal pain due to cholelithiasis. Her father died at the age "
    "of 60 with some form of abdominal cancer. Her temperature is 37\u00b0C (98.6\u00b0F), "
    "respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. "
    "Physical exam is unremarkable. However, a CT scan of the abdomen shows a "
    "calcified mass near her gallbladder. Which of the following diagnoses should "
    "be excluded first in this patient?",
    "A 67-year-old man who was diagnosed with arthritis 16 years ago presents "
    "with right knee swelling and pain. His left knee was swollen a few weeks ago, "
    "but now with both joints affected, he has difficulty walking and feels frustrated. "
    "He also has back pain which makes it extremely difficult to move around and be active "
    "during the day. He says his pain significantly improves with rest. He also suffers "
    "from dandruff for which he uses special shampoos. Physical examination is notable "
    "for pitting of his nails. Which of the following is the most likely diagnosis?",
]
batch = {"text": txt}
new_batch = filter(copy(batch))
for original_txt, new_txt in zip(batch["text"], new_batch["text"]):
    print(get_separator())
    print(f">> {original_txt}")
    print(f">> {new_txt}")
