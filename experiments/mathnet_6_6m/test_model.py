"""Fast architecture checks; run with ``python -m unittest ...test_model``."""

from __future__ import annotations

import unittest

import torch

from .config import DEFAULT_MODEL_CONFIG, EXPECTED_PARAMETER_COUNT
from .model import MathNetGPT


class MathNetGPTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = MathNetGPT(DEFAULT_MODEL_CONFIG).eval()

    def test_parameter_budget_is_exact(self) -> None:
        self.assertEqual(self.model.parameter_count, EXPECTED_PARAMETER_COUNT)

    def test_cached_logits_match_full_logits(self) -> None:
        input_ids = torch.randint(0, DEFAULT_MODEL_CONFIG.vocab_size, (2, 17))
        full_logits, _ = self.model(input_ids)
        cache = None
        pieces = []
        for index in range(input_ids.size(1)):
            logits, cache = self.model(input_ids[:, index : index + 1], cache, use_cache=True)
            pieces.append(logits)
        cached_logits = torch.cat(pieces, dim=1)
        torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=2e-4)

    def test_generation_appends_requested_tokens(self) -> None:
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        generated = self.model.generate(prompt, max_new_tokens=4, top_k=1)
        self.assertEqual(tuple(generated.shape), (1, 7))
        torch.testing.assert_close(generated[:, :3], prompt)


if __name__ == "__main__":
    unittest.main()
