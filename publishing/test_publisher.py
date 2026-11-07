"""Offline regression tests for the release publisher. No live GitHub writes."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "publish.py"

def definitions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    # Load helpers and constants without evaluating command-line publication.
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    names = {"OWNER", "NAME", "COLLABORATOR", "TARGET", "URL", "EXCLUDED", "ENV_PREFIXES",
             "EXCLUDED_DIRECTORY_SUFFIXES", "BUILD_SUFFIXES", "SECRET_MARKERS"}
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            nodes.append(n)
    ns = {"__name__": "publisher_tests"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns

class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.p = definitions()
    def test_empty_transport_result_passes(self):
        command = Mock(return_value=subprocess.CompletedProcess([], 0, b"", b""))
        self.p["command"] = command
        self.p["require_empty"]()
        command.assert_called_once_with(["git", "ls-remote", "--refs", self.p["URL"] + ".git"])
    def test_whitespace_empty_result_passes(self):
        self.p["command"] = Mock(return_value=subprocess.CompletedProcess([], 0, b"\r\n", b""))
        self.p["require_empty"]()
    def test_main_blocks(self):
        self.p["command"] = Mock(return_value=subprocess.CompletedProcess([], 0, b"a"*40+b"\trefs/heads/main\n", b""))
        with self.assertRaisesRegex(RuntimeError, "already contains Git references"):
            self.p["require_empty"]()
    def test_other_branch_blocks(self):
        self.p["command"] = Mock(return_value=subprocess.CompletedProcess([], 0, b"a"*40+b"\trefs/heads/work\n", b""))
        with self.assertRaises(RuntimeError): self.p["require_empty"]()
    def test_tag_only_blocks(self):
        self.p["command"] = Mock(return_value=subprocess.CompletedProcess([], 0, b"a"*40+b"\trefs/tags/v1\n", b""))
        with self.assertRaises(RuntimeError): self.p["require_empty"]()
    def test_transport_failure_blocks(self):
        self.p["command"] = Mock(side_effect=RuntimeError("transport failed"))
        with self.assertRaisesRegex(RuntimeError, "transport failed"): self.p["require_empty"]()
    def test_regression_null_shape_is_irrelevant(self):
        self.assertNotIn('defaultBranchRef', SOURCE.read_text())
        old = lambda v: v.get("isEmpty") is True and v.get("defaultBranchRef") is None
        self.assertTrue(old({"isEmpty": True, "defaultBranchRef": None}))
        self.assertFalse(old({"isEmpty": True, "defaultBranchRef": {"name": ""}}))
    def test_collect_excludes_environment_raw_and_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ["keep.txt", ".git/config", ".venv-report/bin/thing", "data/raw/example", "development/outputs/demo"]:
                f=root/name; f.parent.mkdir(parents=True, exist_ok=True); f.write_text("example")
            manifest, count=self.p["collect"](root)
            self.assertEqual(set(manifest), {"keep.txt"})
            self.assertEqual(count, 4)
    def test_credential_filename_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/".env").write_text("EXAMPLE=1\n")
            with self.assertRaisesRegex(RuntimeError,"credential"): self.p["collect"](root)
    def test_symlink_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"regular").write_text("example")
            try: (root/"linked").symlink_to(root/"regular")
            except OSError: self.skipTest("Symlinks unavailable in this environment")
            with self.assertRaisesRegex(RuntimeError,"symbolic link"): self.p["collect"](root)
    def test_real_empty_and_populated_local_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); remote=base/"remote.git"; work=base/"work"
            subprocess.run(["git","init","--bare",str(remote)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.p["URL"]=str(remote)[:-4]
            def run(argv):
                return subprocess.run(argv,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.p["command"]=run
            self.p["require_empty"]()
            subprocess.run(["git","init","-b","main",str(work)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (work/"README.md").write_text("fixture\n")
            def git(*args):
                return subprocess.run(["git","-c","user.name=Fixture","-c","user.email=fixture@example.invalid",*args],cwd=work,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            git("add","README.md"); git("commit","-m","Fixture"); git("push",str(remote),"main")
            with self.assertRaises(RuntimeError): self.p["require_empty"]()
    def test_wrappers_parse_in_bash(self):
        for name in ("publish.sh","publish.zsh"):
            subprocess.run(["bash","-n",str(HERE/name)],check=True)

if __name__ == "__main__":
    unittest.main(verbosity=2)
