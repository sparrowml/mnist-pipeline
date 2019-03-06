import os
import uuid
import unittest

from .config import MnistConfig


class TestConfig(unittest.TestCase):
    def test_from_yaml__overrides_attributes(self):
        filename = f'.{str(uuid.uuid4())}.yaml'
        with open(filename, 'w') as outfile:
            outfile.write('batch_size: 1')
        config = MnistConfig.from_yaml(filename)
        self.assertEqual(config.batch_size, 1)
        os.remove(filename)

    def test_image_shape__is_image_shape(self):
        config = MnistConfig(
            image_height=20,
            image_width=20,
            n_channels=3
        )
        self.assertEqual(config.image_shape, (20, 20, 3))

    def test_artifact_directory_path__uses_environment_variable(self):
        env = {'ARTIFACT_DIRECTORY': 'foo'}
        with unittest.mock.patch.dict('os.environ', env):
            config = MnistConfig()
            self.assertEqual(str(config.artifact_directory_path), 'foo')
