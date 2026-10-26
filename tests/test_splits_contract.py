"""Split-contract tests for the paper path (no data, no cache)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.config import load_config  # noqa: E402
from framework.data import tail_mask  # noqa: E402
from framework.splits import holdout_dyads, month_idx, temporal_masks  # noqa: E402


class TestTemporalContract(unittest.TestCase):
    def test_month_index_zero_is_1995_01(self):
        self.assertEqual(month_idx("1995-01-01"), 0)
        self.assertEqual(month_idx("2015-01-01"), 240)
        self.assertEqual(month_idx("2016-01-01"), 252)
        self.assertEqual(month_idx("2023-01-01"), 336)

    def test_frozen_split_boundaries(self):
        cfg = load_config()
        t = np.array([0, 239, 240, 251, 252, 275, 276, 335, 336])
        masks = temporal_masks(t, cfg)
        self.assertEqual(t.tolist(), [0, 239, 240, 251, 252, 275, 276, 335, 336])
        self.assertEqual(masks["train"].tolist(),
                         [True, True, False, False, False, False, False, False, False])
        self.assertEqual(masks["select"].tolist(),
                         [False, False, True, True, False, False, False, False, False])
        self.assertEqual(masks["eval"].tolist(),
                         [False, False, False, False, True, True, False, False, False])
        self.assertEqual(masks["val"].tolist(),
                         [False, False, True, True, True, True, False, False, False])
        self.assertEqual(masks["test"].tolist(),
                         [False, False, False, False, False, False, True, True, True])
        self.assertEqual(masks["tail"].tolist(),
                         [False, False, False, False, False, False, False, False, True])

    def test_split_masks_do_not_overlap(self):
        cfg = load_config()
        t = np.arange(0, 360)
        masks = temporal_masks(t, cfg)
        train = masks["train"]
        select = masks["select"]
        evaluate = masks["eval"]
        self.assertFalse(np.any(train & select))
        self.assertFalse(np.any(train & evaluate))
        self.assertFalse(np.any(select & evaluate))
        self.assertTrue(np.all(masks["val"] == (select | evaluate)))

    def test_tail_mask_uses_2023_boundary(self):
        cfg = load_config()
        self.assertFalse(tail_mask(np.array([335]), cfg)[0])
        self.assertTrue(tail_mask(np.array([336]), cfg)[0])


class TestDyadHoldout(unittest.TestCase):
    def _train(self):
        # did 0-4 have at least one positive; did 5-9 are ever-active but never
        # positive in train. The holdout must draw from both strata.
        did = np.repeat(np.arange(10), 4)
        y = np.zeros(len(did), dtype=int)
        y[(did <= 4) & (np.tile(np.arange(4), 10) == 0)] = 1
        return {"did": did, "y": y}

    def test_holdout_is_deterministic_and_stratified(self):
        train = self._train()
        first = holdout_dyads(train, frac=0.2, seed=20261023)
        second = holdout_dyads(train, frac=0.2, seed=20261023)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)  # 1 active + 1 never-active per 5-dyad pool
        active = {0, 1, 2, 3, 4}
        inactive = {5, 6, 7, 8, 9}
        self.assertTrue(first & active)
        self.assertTrue(first & inactive)
        self.assertTrue(first.issubset(active | inactive))

    def test_holdout_seed_changes_membership(self):
        train = self._train()
        first = holdout_dyads(train, frac=0.4, seed=1)
        second = holdout_dyads(train, frac=0.4, seed=2)
        self.assertNotEqual(first, second)

    def test_holdout_never_exceeds_pool(self):
        train = self._train()
        held = holdout_dyads(train, frac=1.0, seed=3)
        self.assertEqual(held, set(range(10)))


class TestActorRows(unittest.TestCase):
    def test_actor_rows_uses_integer_actor_ids(self):
        from framework.splits import actor_rows

        data = {
            "y": np.zeros(4, dtype=int),
            "sid": np.array([1, 2, 3, 4]),
            "tid": np.array([2, 3, 4, 1]),
        }
        mask = actor_rows(data, {2})
        self.assertEqual(mask.tolist(), [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
