from unittest import TestCase

from fz_openqa.datamodules.pipes.passage import gen_passages


class TestPassage(TestCase):
    def test_gen_passages(self):
        configs = [
            {"seq_length": 100, "size": 10, "stride": 7, "pad_token": None},
            {"seq_length": 99, "size": 10, "stride": 7, "pad_token": None},
            {"seq_length": 100, "size": 10, "stride": 10, "pad_token": None},
            {"seq_length": 100, "size": 10, "stride": 8, "pad_token": None},
            {"seq_length": 100, "size": 10, "stride": 10, "pad_token": -1},
            {"seq_length": 100, "size": 10, "stride": 8, "pad_token": "[PAD]"},
            {
                "seq_length": 100,
                "size": 10,
                "stride": 10,
                "pad_token": -1,
                "start_tokens": ["<cls>"],
            },
            {
                "seq_length": 100,
                "size": 10,
                "stride": 8,
                "pad_token": "[PAD]",
                "start_tokens": [1, 2],
            },
        ]
        for cfg in configs:
            self._test_gen_windows(**cfg)

    def _test_gen_windows(self, *, seq_length: int, verbose=False, **kwargs):
        x = list(range(seq_length))
        if verbose:
            print(f"\n> Input: {x}")
        tokens = []
        outputs = []
        for w, m in gen_passages(x, **kwargs):
            outputs += [(w, m)]
            if verbose:
                print([ww if mm else "*" for ww, mm in zip(w, m)], len(w), len(m))
            tokens += [ww for ww, mm in zip(w, m) if mm > 0]

        # test that all input tokens (x) are placed once and only once in the windows
        self.assertTrue(all([xx == tt for xx, tt in zip(x, tokens)]))

        # test that all windows and masks are of the same size
        self.assertTrue(all(len(w) == len(m) for w, m in outputs))

        # test that all sequences are of the same size if a pad_token is provided
        if kwargs["pad_token"] is not None:
            self.assertTrue(all(len(w) == kwargs["size"] for w, m in outputs))
            self.assertTrue(all(len(m) == kwargs["size"] for w, m in outputs))
