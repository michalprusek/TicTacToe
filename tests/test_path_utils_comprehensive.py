# @generated [partially] Claude Code 2025-01-01: Comprehensive tests for updated path_utils
"""
Comprehensive tests for the updated app.main.path_utils module.
Tests all new pathlib-based functions and cross-platform compatibility.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, Mock

from app.main.path_utils import (
    setup_project_path,
    setup_uarm_sdk_path,
    get_project_root,
    get_weights_path,
    get_calibration_path,
    get_calibration_file_path,
    get_detection_model_path,
    get_pose_model_path,
    get_project_root_str,
    get_weights_path_str,
    get_calibration_path_str
)


class TestPathUtilsComprehensive:
    """Comprehensive tests for path_utils module."""

    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.is_absolute()

    def test_get_project_root_consistency(self):
        """Test that get_project_root returns consistent results."""
        root1 = get_project_root()
        root2 = get_project_root()
        assert root1 == root2

    def test_setup_project_path_returns_path(self):
        """Test that setup_project_path returns a Path object."""
        with patch('sys.path') as mock_path:
            mock_path.__contains__ = Mock(return_value=False)
            mock_path.insert = Mock()
            
            root = setup_project_path()
            assert isinstance(root, Path)
            assert root.is_absolute()

    def test_setup_project_path_adds_to_sys_path(self):
        """Test that setup_project_path adds project root to sys.path."""
        with patch('sys.path') as mock_path:
            mock_path.__contains__ = Mock(return_value=False)
            mock_path.insert = Mock()
            
            root = setup_project_path()
            mock_path.insert.assert_called_once_with(0, str(root))

    def test_setup_project_path_skips_if_already_in_path(self):
        """Test that setup_project_path skips adding if already in sys.path."""
        with patch('sys.path') as mock_path:
            mock_path.__contains__ = Mock(return_value=True)
            mock_path.insert = Mock()
            
            setup_project_path()
            mock_path.insert.assert_not_called()

    def test_get_weights_path_returns_path(self):
        """Test that get_weights_path returns a Path object."""
        weights_path = get_weights_path()
        assert isinstance(weights_path, Path)
        assert weights_path.name == "weights"

    def test_get_calibration_path_returns_path(self):
        """Test that get_calibration_path returns a Path object."""
        cal_path = get_calibration_path()
        assert isinstance(cal_path, Path)
        assert cal_path.name == "calibration"

    def test_get_calibration_file_path_returns_path(self):
        """Test that get_calibration_file_path returns a Path object."""
        cal_file_path = get_calibration_file_path()
        assert isinstance(cal_file_path, Path)
        assert cal_file_path.name == "hand_eye_calibration.json"
        assert cal_file_path.suffix == ".json"

    def test_get_detection_model_path_returns_path(self):
        """Test that get_detection_model_path returns a Path object."""
        model_path = get_detection_model_path()
        assert isinstance(model_path, Path)
        assert model_path.name == "best_detection.pt"
        assert model_path.suffix == ".pt"

    def test_get_pose_model_path_returns_path(self):
        """Test that get_pose_model_path returns a Path object."""
        model_path = get_pose_model_path()
        assert isinstance(model_path, Path)
        assert model_path.name == "best_pose.pt"
        assert model_path.suffix == ".pt"

    def test_path_relationships(self):
        """Test that paths have correct relationships."""
        root = get_project_root()
        weights = get_weights_path()
        calibration = get_calibration_path()
        cal_file = get_calibration_file_path()
        detection_model = get_detection_model_path()
        pose_model = get_pose_model_path()

        # Check parent relationships
        assert weights.parent == root
        assert calibration.parent == root / "app"
        assert cal_file.parent == calibration
        assert detection_model.parent == weights
        assert pose_model.parent == weights

    def test_setup_uarm_sdk_path_returns_none_if_not_exists(self):
        """Test that setup_uarm_sdk_path returns None if SDK doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            result = setup_uarm_sdk_path()
            assert result is None

    def test_setup_uarm_sdk_path_returns_path_if_exists(self):
        """Test that setup_uarm_sdk_path returns Path if SDK exists."""
        with patch.object(Path, 'exists', return_value=True), \
             patch('sys.path') as mock_path:
            mock_path.__contains__ = Mock(return_value=False)
            mock_path.insert = Mock()
            
            result = setup_uarm_sdk_path()
            assert isinstance(result, Path)
            assert result.name == "uArm-Python-SDK"

    def test_setup_uarm_sdk_path_adds_to_sys_path_if_exists(self):
        """Test that setup_uarm_sdk_path adds SDK to sys.path if exists."""
        with patch.object(Path, 'exists', return_value=True), \
             patch('sys.path') as mock_path:
            mock_path.__contains__ = Mock(return_value=False)
            mock_path.insert = Mock()
            
            result = setup_uarm_sdk_path()
            mock_path.insert.assert_called_once_with(0, str(result))

    def test_backward_compatibility_functions_return_strings(self):
        """Test that backward compatibility functions return strings."""
        root_str = get_project_root_str()
        weights_str = get_weights_path_str()
        cal_str = get_calibration_path_str()

        assert isinstance(root_str, str)
        assert isinstance(weights_str, str)
        assert isinstance(cal_str, str)

    def test_backward_compatibility_consistency(self):
        """Test that backward compatibility functions match Path versions."""
        assert get_project_root_str() == str(get_project_root())
        assert get_weights_path_str() == str(get_weights_path())
        assert get_calibration_path_str() == str(get_calibration_path())

    def test_paths_are_absolute(self):
        """Test that all returned paths are absolute."""
        paths = [
            get_project_root(),
            get_weights_path(),
            get_calibration_path(),
            get_calibration_file_path(),
            get_detection_model_path(),
            get_pose_model_path()
        ]
        
        for path in paths:
            assert path.is_absolute(), f"Path {path} is not absolute"

    def test_cross_platform_compatibility(self):
        """Test that paths work across different platforms."""
        # Test that paths use forward slashes internally (pathlib handles this)
        root = get_project_root()
        weights = get_weights_path()
        
        # Convert to string and check that it's platform appropriate
        root_str = str(root)
        weights_str = str(weights)
        
        # On all platforms, pathlib should handle separators correctly
        assert len(root_str) > 0
        assert len(weights_str) > 0
        assert weights_str.endswith("weights")

    def test_path_resolution(self):
        """Test that paths are properly resolved."""
        root = get_project_root()
        
        # Should be resolved (no .. or . components)
        assert ".." not in str(root)
        assert "/." not in str(root) and "\\." not in str(root)

    @pytest.mark.parametrize("path_func,expected_name", [
        (get_weights_path, "weights"),
        (get_calibration_path, "calibration"),
        (get_calibration_file_path, "hand_eye_calibration.json"),
        (get_detection_model_path, "best_detection.pt"),
        (get_pose_model_path, "best_pose.pt"),
    ])
    def test_path_names(self, path_func, expected_name):
        """Test that path functions return paths with correct names."""
        path = path_func()
        assert path.name == expected_name

    def test_path_types_consistency(self):
        """Test that all path functions return Path objects."""
        path_functions = [
            get_project_root,
            get_weights_path,
            get_calibration_path,
            get_calibration_file_path,
            get_detection_model_path,
            get_pose_model_path
        ]
        
        for func in path_functions:
            result = func()
            assert isinstance(result, Path), f"{func.__name__} should return Path"

    def test_string_functions_consistency(self):
        """Test that string functions return strings."""
        string_functions = [
            get_project_root_str,
            get_weights_path_str,
            get_calibration_path_str
        ]
        
        for func in string_functions:
            result = func()
            assert isinstance(result, str), f"{func.__name__} should return str"
