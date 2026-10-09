"""Module-local setup checks; no installation, network access or GPU use."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

MODULE = Path(__file__).resolve().parents[1]
SCRIPT = MODULE / 'scripts/setup_external.sh'


class SetupTests(unittest.TestCase):
    def fixture(self, base):
        # A different module name and unrelated parent exercise portability after merging.
        root = Path(base) / "parent with spaces" / "renamed module"
        (root / 'scripts').mkdir(parents=True)
        shutil.copy2(SCRIPT, root / 'scripts/setup_external.sh')
        (root / 'capsim/third_party/RoboTwin/envs').mkdir(parents=True)
        return root

    def invoke(self, root, *args, **kwargs):
        return subprocess.run(['bash', str(root / 'scripts/setup_external.sh'), *args],
                              env={**os.environ, 'SETUP_PYTHON': sys.executable, **kwargs.pop('env', {})},
                              capture_output=True, text=True, **kwargs)

    def test_configuration_is_module_local_and_preserves_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            site = root / 'site.yaml'
            old = {'data_root': '/custom/data', 'robotwin_py': '/old/python'}
            site.write_text(yaml.safe_dump(old))
            trainer = Path(temporary) / 'trainer'
            slicer = trainer / 'robotwin/dataprep/slicing.py'
            slicer.parent.mkdir(parents=True); slicer.write_text('')
            args = ('configure', '--trainer-root', str(trainer), '--train-python', sys.executable)
            self.invoke(root, *args, check=True)
            content = site.read_text()
            self.invoke(root, *args, check=True)
            self.assertEqual(content, site.read_text())
            actual = yaml.safe_load(content)
            self.assertEqual(actual['data_root'], old['data_root'])
            self.assertEqual(actual['robotwin_py'], str(Path(sys.executable).resolve()))
            self.assertEqual(actual['slicer'], str(slicer))
            self.assertTrue(actual['robotwin_root'].startswith(str(root)))
            self.assertEqual(yaml.safe_load((root / 'local/config/site.yaml.before-setup').read_text()), old)
            self.assertFalse((root.parent / 'site.yaml').exists())

    def test_incomplete_trainer_arguments_do_not_change_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            site = root / 'site.yaml'; site.write_text('data_root: /keep\n')
            result = self.invoke(root, 'configure', '--trainer-root', temporary)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('both', result.stderr)
            self.assertEqual(site.read_text(), 'data_root: /keep\n')
            self.assertFalse((root / 'local/config/site.yaml.before-setup').exists())

    def test_parent_git_metadata_cannot_identify_external_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            bin_dir = Path(temporary) / 'bin'; bin_dir.mkdir()
            nvcc = bin_dir / 'nvcc'; nvcc.write_text('#!/bin/sh\nexit 0\n'); nvcc.chmod(0o755)
            result = self.invoke(root, 'robotwin', env={
                'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'], 'SETUP_PYTHON': '/bin/true'})
            self.assertEqual(result.returncode, 2)
            self.assertIn('independent Git checkout', result.stderr)
            self.assertFalse((root / 'capsim/third_party/RoboTwin/envs/curobo').exists())

    def test_help_and_unknown_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            self.assertEqual(self.invoke(root, '--help').returncode, 0)
            self.assertEqual(self.invoke(root, 'unknown').returncode, 2)
            self.assertFalse((root / 'site.yaml').exists())


if __name__ == '__main__':
    unittest.main()
