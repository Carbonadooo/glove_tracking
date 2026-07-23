import json
import tempfile
import unittest
from pathlib import Path

from hand_restoration.data_config import resolve_clip_splits


class DataConfigTests(unittest.TestCase):
    def test_legacy_clip_tars_remain_supported(self):
        train, val, split = resolve_clip_splits({"data": {"clip_tars": ["a.tar"]}})
        self.assertEqual(train, ["a.tar"])
        self.assertEqual(val, [])
        self.assertIsNone(split)

    def test_split_manifest_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(json.dumps({"train": ["a.tar"], "holdout": ["b.tar"]}))
            train, val, resolved = resolve_clip_splits({"data": {"split_json": str(split)}}, root)
            self.assertEqual((train, val, resolved), (["a.tar"], ["b.tar"], split))
            split.write_text(json.dumps({"train": ["dir/a.tar"], "holdout": ["other/a.tar"]}))
            with self.assertRaisesRegex(ValueError, "leakage"):
                resolve_clip_splits({"data": {"split_json": str(split)}}, root)


if __name__ == "__main__":
    unittest.main()
