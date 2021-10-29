from unittest import TestCase

import numpy as np
import torch

from fz_openqa.modeling.functional import count_right_padding, pad, _padless_cat


class Test_functional(TestCase):
    def test_padless_cat_case(self):
        a = {"input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 0, 3, 4, 5]])}
        b = {"input_ids": torch.tensor([[4, 0, 0], [6, 7, 8]])}
        for x in [a, b]:
            x["attention_mask"] = x["input_ids"].clone()
        ab = _padless_cat(a, b, pad_token=0, aux_pad_tokens={"attention_mask": -1})
        ref = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0, 0], [1, 2, 0, 3, 4, 5, 6, 7, 8]])
        attn_ref = torch.tensor([[1, 2, 3, 4, 0, 0, -1, -1, -1], [1, 2, 0, 3, 4, 5, 6, 7, 8]])
        self.assertTrue((1 - (ab["input_ids"] == ref).float()).sum() == 0)
        self.assertTrue((1 - (ab["attention_mask"] == attn_ref).float()).sum() == 0)

    def test_padless_cat_generated(self):
        length = 10
        bs = 3
        n_trials = 10

        def count(x, value):
            return [t for t in x.view(-1)].count(value)

        def gen_seqs(token, pad_token, length):
            ls = np.random.randint(1, length - 1)
            return ls * [token] + (length - ls) * [pad_token]

        for k in range(n_trials):
            for pad_tok in [0, -100]:
                a = {"input_ids": torch.tensor([gen_seqs(1, pad_tok, length) for _ in range(bs)])}
                b = {"input_ids": torch.tensor([gen_seqs(2, pad_tok, length) for _ in range(bs)])}
                for x in [a, b]:
                    x["attention_mask"] = x["input_ids"].clone()
                    x["idx"] = torch.ones_like(x["input_ids"][:, 0])

                # concatenate
                ab = _padless_cat(a, b, pad_token=pad_tok)

                self.assertEqual(a.keys(), ab.keys())
                self.assertEqual(a.keys(), b.keys())
                self.assertTrue(all(xab == xa for xab, xa in zip(ab["idx"], a["idx"])))
                for xa, xb, xab, xab_atn in zip(
                    a["input_ids"], b["input_ids"], ab["input_ids"], ab["attention_mask"]
                ):
                    self.assertEqual(count(xa, 1), count(xab, 1))
                    self.assertEqual(count(xb, 2), count(xab, 2))
                    self.assertLess(len(xab), len(xa) + len(xb))
                    self.assertEqual(len(xab), len(xab_atn))

    def test_pad(self):
        batch = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4, 0, 0])]
        padded = pad(batch, 0)
        self.assertEqual(padded.shape, torch.Size([2, 4]))

        batch = [torch.tensor([1, 2, 3, 0, 0]), torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])]
        padded = pad(batch, 0)
        self.assertEqual(padded.shape, torch.Size([2, 4]))

        batch = [torch.tensor([1, 2, 3, 0, 0]), torch.tensor([0, 1, 2, 3, 4, 0, 0, 0, 0])]
        padded = pad(batch, 0)
        self.assertEqual(padded.shape, torch.Size([2, 5]))

    def test_count_right_padding(self):
        x = [1, 2, 3]
        self.assertEqual(count_right_padding(x, 0), 0)

        x = [1, 2, 3, 0, 0]
        print(count_right_padding(x, 0))
        self.assertEqual(count_right_padding(x, 0), 2)

        x = [1, 2, 0, 3, "x", "x"]
        self.assertEqual(count_right_padding(x, "x"), 2)

        x = [0, 1, 2, 0, 3]
        self.assertEqual(count_right_padding(x, 0), 0)
