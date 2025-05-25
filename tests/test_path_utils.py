"""
Tests for path_utils module.
"""
import pytest
import sys
import os
from unittest.mock import patch

from app.main.path_utils import (
    setup_project_path, setup_uarm_sdk_path, get_project_root,
    get_weights_path, get_calibration_path
)


class TestPathUtilities:
    """Test class for path utility functions."""
    
    def test_setup_project_path(self):
        """Test project path setup."""
        # Save original sys.path
        original_path = sys.path.copy()
        
        try:
            # Remove any existing project paths for clean test
            root = setup_project_path()
            
            # Check that path was added
            assert root in sys.path
            assert os.path.isabs(root)  # Should be absolute path
            assert root.endswith('TicTacToe')  # Should end with project name
            
        finally:
            # Restore original sys.path
            sys.path[:] = original_path
    
    def test_get_project_root(self):
        """Test getting project root directory."""
        root = get_project_root()
        
        assert os.path.isabs(root)  # Should be absolute path
        assert os.path.exists(root)  # Should exist
        assert root.endswith('TicTacToe')  # Should end with project name
    
    def test_get_weights_path(self):
        """Test getting weights directory path."""
        weights_path = get_weights_path()
        
        assert os.path.isabs(weights_path)  # Should be absolute path
        assert weights_path.endswith('weights')  # Should end with weights
        # Note: We don't test existence as weights dir might not exist in test env    
    def test_get_calibration_path(self):
        """Test getting calibration directory path."""
        calib_path = get_calibration_path()
        
        assert os.path.isabs(calib_path)  # Should be absolute path
        assert 'calibration' in calib_path  # Should contain calibration
    
    @patch('os.path.exists')
    def test_setup_uarm_sdk_path_exists(self, mock_exists):
        """Test uArm SDK path setup when SDK exists."""
        original_path = sys.path.copy()
        
        try:
            mock_exists.return_value = True
            
            sdk_path = setup_uarm_sdk_path()
            
            assert sdk_path is not None
            assert sdk_path in sys.path
            assert 'uArm-Python-SDK' in sdk_path
            
        finally:
            sys.path[:] = original_path
    
    @patch('os.path.exists')
    def test_setup_uarm_sdk_path_not_exists(self, mock_exists):
        """Test uArm SDK path setup when SDK doesn't exist."""
        mock_exists.return_value = False
        
        sdk_path = setup_uarm_sdk_path()
        
        assert sdk_path is None
    
    def test_path_consistency(self):
        """Test that all path functions return consistent roots."""
        project_root = get_project_root()
        setup_root = setup_project_path()
        
        # Both should return the same project root
        assert project_root == setup_root
        
        # Weights and calibration should be under project root
        weights_path = get_weights_path()
        calib_path = get_calibration_path()
        
        assert weights_path.startswith(project_root)
        assert calib_path.startswith(project_root)