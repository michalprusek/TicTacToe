"""
Tests for path_utils module.
"""
import unittest
import os
import sys
from unittest.mock import patch

from app.main.path_utils import (
    setup_project_path, setup_uarm_sdk_path, get_project_root,
    get_weights_path, get_calibration_path
)


class TestPathUtils(unittest.TestCase):
    
    def setUp(self):
        self.original_path = sys.path.copy()
    
    def tearDown(self):
        sys.path[:] = self.original_path
    
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    def test_setup_project_path(self, mock_abspath, mock_dirname):
        mock_abspath.return_value = "/fake/app/main/path_utils.py"
        mock_dirname.side_effect = ["/fake/app/main", "/fake/app", "/fake"]
        
        sys.path.clear()
        result = setup_project_path()
        self.assertEqual(result, "/fake")
        self.assertIn("/fake", sys.path)
    
    @patch('os.path.exists')
    @patch('os.path.dirname') 
    @patch('os.path.abspath')
    def test_uarm_sdk_exists(self, mock_abspath, mock_dirname, mock_exists):
        mock_abspath.return_value = "/fake/app/main/path_utils.py"
        mock_dirname.side_effect = ["/fake/app/main", "/fake/app", "/fake"]
        mock_exists.return_value = True
        
        result = setup_uarm_sdk_path()
        self.assertEqual(result, "/fake/uArm-Python-SDK")
    
    @patch('os.path.exists')
    def test_uarm_sdk_not_exists(self, mock_exists):
        mock_exists.return_value = False
        result = setup_uarm_sdk_path()
        self.assertIsNone(result)    
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    def test_get_project_root(self, mock_abspath, mock_dirname):
        mock_abspath.return_value = "/fake/app/main/path_utils.py"
        mock_dirname.side_effect = ["/fake/app/main", "/fake/app", "/fake"]
        
        result = get_project_root()
        self.assertEqual(result, "/fake")
    
    @patch('os.path.join')
    @patch('app.main.path_utils.get_project_root')
    def test_get_weights_path(self, mock_get_root, mock_join):
        mock_get_root.return_value = "/fake"
        mock_join.return_value = "/fake/weights"
        
        result = get_weights_path()
        self.assertEqual(result, "/fake/weights")
        mock_join.assert_called_once_with("/fake", "weights")
    
    @patch('os.path.join')
    @patch('app.main.path_utils.get_project_root')
    def test_get_calibration_path(self, mock_get_root, mock_join):
        mock_get_root.return_value = "/fake"
        mock_join.return_value = "/fake/app/calibration"
        
        result = get_calibration_path()
        self.assertEqual(result, "/fake/app/calibration")
        mock_join.assert_called_once_with("/fake", "app", "calibration")


if __name__ == '__main__':
    unittest.main()