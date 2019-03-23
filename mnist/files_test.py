import uuid
import unittest
from pathlib import Path

from .files import MnistFiles


class MockFiles(MnistFiles):
    _artifact_url_prefix: str = 'FAKE'


class TestFiles(unittest.TestCase):
    """
    This assumes the save-datasets command has been run
    """
    def test_constructor__creates_directory(self):
        identifier = str(uuid.uuid4())
        directory = f'./.{identifier}'
        env = {'ARTIFACT_DIRECTORY': directory}
        with unittest.mock.patch.dict('os.environ', env):
            _ = MockFiles()
        directory = Path(directory)
        self.assertTrue(directory.exists())
        directory.rmdir()

    def test_train_dataset__exists(self):
        files = MockFiles()
        self.assertTrue(Path(files.train_dataset).exists())

    def test_test_dataset__exists(self):
        files = MockFiles()
        self.assertTrue(Path(files.test_dataset).exists())

    def test_model_weights__exists(self):
        files = MockFiles()
        self.assertTrue(Path(files.model_weights).exists())

    def test_download_model_weights__skips_download(self):
        # We should short circuit the download since the file exists
        files = MockFiles()
        self.assertIsNotNone(files.download_model_weights())

    def test_feature_weights__exists(self):
        files = MockFiles()
        self.assertTrue(Path(files.feature_weights).exists())

    def test_download_feature_weights__skips_download(self):
        # We should short circuit the download since the file exists
        files = MockFiles()
        self.assertIsNotNone(files.download_feature_weights())
