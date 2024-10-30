import unittest
from mock import patch, MagicMock
from src.model.zarr.foo import foo


class TestFoo(unittest.TestCase):

    # @patch("model.zarr.foo.Bar.biz")  # not -> @patch("bar.Bar.biz")
    # def test_foo(self, mock_biz):
    #     self.assertFalse(mock_biz.called)
    #
    #     foo()
    #
    #     self.assertTrue(mock_biz.called)
    #     self.assertEqual(mock_biz.call_count, 1)
    #     self.assertIsInstance(mock_biz, MagicMock)

    # @patch("model.zarr.bar.requests.get")
    # @patch("model.zarr.bar.requests.put")
    # def test_foo(self, mock_put, mock_get):
    #     bar = Bar()
    #     bar.sync(id=42, query_first=False)
    #
    #     self.assertFalse(mock_get.called)
    #
    #     self.assertTrue(mock_put.called)
    #
    #     bar.sync(id=43, query_first=True)
    #
    #     self.assertTrue(mock_get.called)

    # @patch("model.zarr.foo.Bar.biz")
    # def test_foo(self, mock_biz):
    #     url = '/api/users/{id}'.format(id=1)
    #     data = {'phone_number': '+17025551000'}
    #     method = 'PUT'
    #     headers = {"Authorization": "JWT <your_token>"}
    #
    #     foo(url, method, data=data, headers=headers)
    #
    #     self.assertTrue(mock_biz.called)
    #     self.assertEqual(mock_biz.call_count, 1)
    #     self.assertEqual(mock_biz.call_args[0][0], url)
    #     self.assertEqual(mock_biz.call_args[0][1], method)
    #     self.assertEqual(mock_biz.call_args[1]['data'], data)
    #     self.assertEqual(mock_biz.call_args[1]['headers'], headers)


    def test_foo(self):
        ret = foo()
        self.assertEqual(ret, 1)

    @patch("model.zarr.foo.Bar.biz")
    def test_foo(self, mock_biz):
        expected_value = 2
        mock_biz.return_value = expected_value

        ret = foo()

        self.assertEqual(ret, expected_value)