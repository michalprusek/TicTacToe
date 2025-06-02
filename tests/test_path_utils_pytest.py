# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Pure pytest tests for path utils module.
"""
import os
from pathlib import Path
from app.main.path_utils import (
    setup_project_path, setup_uarm_sdk_path, get_project_root,
    get_weights_path, get_calibration_path
)


class TestPathUtils:
    """Pure pytest test class for path utils."""

    def test_get_project_root(self):
        """Test getting project root directory."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        assert root.is_dir()

    def test_setup_project_path(self):
        """Test setting up project path."""
        import sys
        original_path = sys.path.copy()
        root = setup_project_path()
        assert isinstance(root, Path)
        assert root.exists()
        assert str(root) in sys.path
        sys.path[:] = original_path

    def test_get_weights_path(self):
        """Test getting weights directory path."""
        weights_path = get_weights_path()
        assert isinstance(weights_path, Path)
        assert weights_path.name == "weights"

    def test_get_calibration_path(self):
        """Test getting calibration path."""
        calibration_path = get_calibration_path()
        assert isinstance(calibration_path, Path)
        assert calibration_path.name == "calibration"

    def test_setup_uarm_sdk_path(self):
        """Test uArm SDK path setup."""
        result = setup_uarm_sdk_path()
        if result is not None:
            assert isinstance(result, Path)
            assert result.exists()

    def test_path_validation(self):
        """Test path validation."""
        root = get_project_root()
        assert root.is_absolute()

        weights = get_weights_path()
        assert weights.is_absolute()
