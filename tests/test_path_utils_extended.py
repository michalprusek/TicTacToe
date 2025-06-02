# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Extended tests for path_utils module.
Pure pytest implementation without unittest.
"""
import pytest
import os
import sys
from unittest.mock import patch, Mock
from pathlib import Path

from app.main.path_utils import (
    get_project_root,
    setup_project_path,
    get_weights_path,
    get_calibration_path,
    setup_uarm_sdk_path
)


class TestPathUtilsExtended:
    """Extended tests for path utilities."""

    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a valid path."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()

    def test_get_project_root_contains_expected_files(self):
        """Test that project root contains expected project files."""
        root = get_project_root()
        # Should contain some project files
        expected_files = ['requirements.txt', 'pytest.ini', 'README.md']
        found_files = [f for f in expected_files if (root / f).exists()]
        assert len(found_files) > 0  # At least one should exist

    @patch('sys.path')
    def test_setup_project_path_adds_to_syspath(self, mock_syspath):
        """Test that setup_project_path adds to sys.path."""
        mock_syspath.insert = Mock()
        setup_project_path()
        mock_syspath.insert.assert_called()

    def test_get_weights_path_returns_path(self):
        """Test that get_weights_path returns a Path object."""
        weights_path = get_weights_path()
        assert isinstance(weights_path, Path)
        # Should end with 'weights'
        assert weights_path.name == 'weights'

    def test_get_calibration_path_returns_path(self):
        """Test that get_calibration_path returns a Path object."""
        cal_path = get_calibration_path()
        assert isinstance(cal_path, Path)
        # Should contain 'calibration' in the path
        assert cal_path.name == 'calibration'

    @patch('sys.path')
    def test_setup_uarm_sdk_path_with_existing_path(self, mock_syspath):
        """Test setup_uarm_sdk_path when path exists."""
        mock_syspath.insert = Mock()

        # Mock Path.exists to return True
        with patch.object(Path, 'exists', return_value=True):
            setup_uarm_sdk_path()
            mock_syspath.insert.assert_called()

    @patch('sys.path')
    def test_setup_uarm_sdk_path_with_nonexistent_path(self, mock_syspath):
        """Test setup_uarm_sdk_path when path doesn't exist."""
        mock_syspath.insert = Mock()

        # Mock Path.exists to return False
        with patch.object(Path, 'exists', return_value=False):
            setup_uarm_sdk_path()
            # Should not add to path if directory doesn't exist
            mock_syspath.insert.assert_not_called()

    def test_paths_are_absolute(self):
        """Test that returned paths are absolute."""
        root = get_project_root()
        weights = get_weights_path()
        calibration = get_calibration_path()

        assert root.is_absolute()
        assert weights.is_absolute()
        assert calibration.is_absolute()

    def test_paths_consistency(self):
        """Test that paths are consistent relative to project root."""
        root = get_project_root()
        weights = get_weights_path()
        calibration = get_calibration_path()

        # Weights should be under project root
        weights_under_root = str(weights).startswith(str(root))
        assert weights_under_root

        # Calibration should be under project root
        calibration_under_root = str(calibration).startswith(str(root))
        assert calibration_under_root

    def test_multiple_calls_return_same_paths(self):
        """Test that multiple calls return the same paths."""
        root1 = get_project_root()
        root2 = get_project_root()
        assert root1 == root2

        weights1 = get_weights_path()
        weights2 = get_weights_path()
        assert weights1 == weights2

        cal1 = get_calibration_path()
        cal2 = get_calibration_path()
        assert cal1 == cal2