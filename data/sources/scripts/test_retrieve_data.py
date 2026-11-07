"""Focused checks for byte identity, path containment, and existing-file protection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('retrieve_data', Path(__file__).with_name('retrieve_data.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.contents = b'fixed historical bytes\n'
        self.entry = {'benchmark': 'fixture', 'local_path': 'tables/fixture/source.parquet',
                      'sha256': hashlib.sha256(self.contents).hexdigest(), 'bytes': len(self.contents)}

    def tearDown(self):
        self.temporary.cleanup()

    def test_matches_and_missing(self):
        path = self.root / 'source'
        self.assertEqual(MODULE.check(path, self.entry), 'missing')
        path.write_bytes(self.contents)
        self.assertEqual(MODULE.check(path, self.entry), 'verified')

    def test_same_length_corruption(self):
        path = self.root / 'source'
        path.write_bytes(b'x' * len(self.contents))
        self.assertEqual(MODULE.check(path, self.entry), 'sha256_mismatch')

    def test_size_mismatch(self):
        path = self.root / 'source'
        path.write_bytes(b'x')
        self.assertEqual(MODULE.check(path, self.entry), 'size_mismatch')

    def test_relative_traversal_and_absolute_paths(self):
        for value in ['../escape', '/absolute', 'safe/../../escape', 'C:\\escape']:
            with self.assertRaises(ValueError):
                MODULE.safe_path(self.root, value)

    def test_symlink_escape(self):
        outside = self.root.parent
        (self.root / 'link').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            MODULE.safe_path(self.root, 'link/escaped')

    def test_existing_conflict_stays_intact(self):
        path = self.root / self.entry['local_path']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'keep these bytes')
        with self.assertRaises(ValueError):
            MODULE.fetch(self.entry, self.root, self.root / 'cache')
        self.assertEqual(path.read_bytes(), b'keep these bytes')

    def test_existing_match_avoids_provider_call(self):
        path = self.root / self.entry['local_path']
        path.parent.mkdir(parents=True)
        path.write_bytes(self.contents)
        self.assertEqual(MODULE.fetch(self.entry, self.root, self.root / 'cache'), 'verified_existing')

    def test_unknown_benchmark_is_rejected(self):
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'file_count': 1, 'files': [self.entry]}))
        with self.assertRaises(ValueError):
            MODULE.records(manifest, ['absent'])


if __name__ == '__main__':
    unittest.main()
