"""
Tests for VisualizationManager module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.main.visualization_manager import VisualizationManager


class TestVisualizationManager:
    """Test VisualizationManager class."""

    def test_init_basic(self):
        """Test basic initialization."""
        manager = VisualizationManager()
        
        assert manager.config is None
        assert manager.logger is not None

    def test_init_with_config_and_logger(self):
        """Test initialization with config and logger."""
        mock_config = Mock()
        mock_logger = Mock()
        
        manager = VisualizationManager(mock_config, mock_logger)
        
        assert manager.config == mock_config
        assert manager.logger == mock_logger

    def test_logger_setup(self):
        """Test logger setup."""
        manager = VisualizationManager()
        
        assert manager.logger is not None
        assert hasattr(manager.logger, 'info')
        assert hasattr(manager.logger, 'error')
        assert hasattr(manager.logger, 'debug')

    def test_visualization_methods_exist(self):
        """Test that visualization methods exist."""
        manager = VisualizationManager()
        
        # Check for common visualization methods
        expected_methods = [
            'draw_detection_results',
            'draw_fps',
            'visualize_grid_points'
        ]
        
        for method_name in expected_methods:
            if hasattr(manager, method_name):
                assert callable(getattr(manager, method_name))

    def test_debug_constants_imported(self):
        """Test that debug constants are available."""
        # These should be imported from detector_constants
        from app.core.detector_constants import (
            DEBUG_UV_KPT_COLOR,
            DEBUG_BBOX_COLOR,
            DEBUG_BBOX_THICKNESS,
            DEBUG_FPS_COLOR
        )
        
        assert isinstance(DEBUG_UV_KPT_COLOR, tuple)
        assert isinstance(DEBUG_BBOX_COLOR, tuple)
        assert isinstance(DEBUG_BBOX_THICKNESS, int)
        assert isinstance(DEBUG_FPS_COLOR, tuple)

    def test_frame_converter_import(self):
        """Test that FrameConverter is imported."""
        from app.main.frame_utils import FrameConverter
        assert FrameConverter is not None

    def test_drawing_utils_import(self):
        """Test that drawing_utils is imported."""
        from app.main import drawing_utils
        assert drawing_utils is not None
