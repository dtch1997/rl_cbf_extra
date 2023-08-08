import unittest

import rl_cbf_extra


class VersionTestCase(unittest.TestCase):
    """Version tests"""

    def test_version(self):
        """check rl_cbf_extra exposes a version attribute"""
        self.assertTrue(hasattr(rl_cbf_extra, "__version__"))
        self.assertIsInstance(rl_cbf_extra.__version__, str)


if __name__ == "__main__":
    unittest.main()
