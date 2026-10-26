"""Feature-contract tests for the paper path (no data, no cache)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework import features as F  # noqa: E402


class TestFeatureLists(unittest.TestCase):
    def test_primary_has_29_dyad_plus_20_graph_features(self):
        self.assertEqual(len(F.DYAD_FEATURES), 29)
        self.assertEqual(len(F.GRAPH_CORE20), 20)
        self.assertEqual(len(F.DYAD_FEATURES + F.GRAPH_CORE20), 49)
        self.assertEqual(len(set(F.DYAD_FEATURES)), 29)
        self.assertEqual(len(set(F.DYAD_FEATURES + F.GRAPH_CORE20)), 49)

    def test_dyad_causal_excludes_train_only_statistics(self):
        self.assertEqual(set(F.DYAD_CAUSAL), set(F.DYAD_FEATURES) - {"br", "seen_tr"})
        self.assertEqual(len(F.DYAD_CAUSAL), 27)

    def test_dyad_groups_partition_the_29_features(self):
        grouped = [name for cols in F.DYAD_GROUPS.values() for name in cols]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), set(F.DYAD_FEATURES))

    def test_graph_core_is_a_subset_of_all_graph_features(self):
        self.assertTrue(set(F.GRAPH_CORE20).issubset(set(F.GRAPH_FEATURES)))
        self.assertEqual(len(F.GRAPH_FEATURES), 35)
        self.assertEqual(len(set(F.GRAPH_FEATURES)), 35)


class TestNoLabelLeakage(unittest.TestCase):
    def test_feature_lists_contain_no_labels(self):
        for names in (F.DYAD_FEATURES, F.GRAPH_FEATURES, F.ACTOR_FEATURES):
            self.assertFalse(set(names) & F.LABELS)
            F.assert_no_label(names)

    def test_assert_no_label_fails_loudly(self):
        with self.assertRaises(AssertionError):
            F.assert_no_label(["active", "y"])

    def test_manifest_is_available_at_t_only(self):
        manifest = F.feature_manifest()
        rows = manifest["dyad"] + manifest["graph"] + manifest["actor"]
        self.assertGreater(len(rows), 49)
        self.assertTrue(all(row["available_at"] == "t" for row in rows))
        names = {row["name"] for row in rows}
        self.assertFalse(names & set(F.LABELS))
        self.assertEqual(manifest["train_only_statistics"], ["br", "seen_tr"])
        self.assertEqual(len(manifest["dyad"]), 29)
        self.assertEqual(len(manifest["graph"]), 35)


if __name__ == "__main__":
    unittest.main()
