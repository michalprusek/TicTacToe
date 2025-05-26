# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Pure pytest tests for path utils module.
"""
import pytest
import os
from app.main.path_utils import (
    setup_project_path, setup_uarm_sdk_path, get_project_root,
    get_weights_path, get_calibration_path
)


class TestPathUtils:
    """Pure pytest test class for path utils."""
    
    def test_get_project_root(self):
        """Test getting project root directory."""
        root = get_project_root()
        assert isinstance(root, str)
        assert os.path.exists(root)
        assert os.path.isdir(root)
    
    def test_setup_project_path(self):
        """Test setting up project path."""
        import sys
        original_path = sys.path.copy()
        root = setup_project_path()
        assert isinstance(root, str)
        assert os.path.exists(root)
        assert root in sys.path
        sys.path[:] = original_path
    
    def test_get_weights_path(self):
        """Test getting weights directory path."""
        weights_path = get_weights_path()
        assert isinstance(weights_path, str)
        assert "weights" in weights_path
        assert weights_path.endswith("weights")
    
    def test_get_calibration_path(self):
        """Test getting calibration path."""
        calibration_path = get_calibration_path()
        assert isinstance(calibration_path, str)
        assert "calibration" in calibration_path
        assert calibration_path.endswith("calibration")
    
    def test_setup_uarm_sdk_path(self):
        """Test uArm SDK path setup."""
        result = setup_uarm_sdk_path()
        if result is not None:
            assert isinstance(result, str)
            assert os.path.exists(result)
    
    def test_path_validation(self):
        """Test path validation."""
        root = get_project_root()
        assert os.path.isabs(root)
        
        weights = get_weights_path()
        assert os.path.isabs(weights)
