import unittest
from mock import patch
from zarr_manager.foo import foo


class TestFoo(unittest.TestCase):
    def test_foo(self):
        ret = foo()
        self.assertEqual(ret, 1)

    @patch("water-column-sonar-processing.zarr_manager.foo.Bar.biz")
    def test_foo(self, mock_biz):
        expected_value = 2
        mock_biz.return_value = expected_value

        ret = foo()

        self.assertEqual(ret, expected_value)