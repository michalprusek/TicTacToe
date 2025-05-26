"""
Comprehensive pytest tests for SymbolDetector module.
Tests symbol detection, configuration, and YOLO model integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2

from app.main.symbol_detector import SymbolDetector


@pytest.fixture
def mock_model():
    """Create mock YOLO model for testing."""
    mock = Mock()
    # Mock typical YOLO output format
    mock.return_value = [Mock()]  # Mock prediction results
    mock.return_value[0].boxes = Mock()
    mock.return_value[0].boxes.xyxy = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
    mock.return_value[0].boxes.conf = np.array([0.8, 0.9])
    mock.return_value[0].boxes.cls = np.array([0, 1])
    return mock


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock()
    config.bbox_conf_threshold = 0.6
    config.class_id_to_label = {0: 'O', 1: 'X'}
    config.class_id_to_player = {0: 1, 1: 2}
    return config


@pytest.fixture
def sample_frame():
    """Create sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestSymbolDetectorInit:
    """Test SymbolDetector initialization."""

    def test_initialization_with_config(self, mock_model, mock_config):
        """Test SymbolDetector initialization with configuration."""
        detector = SymbolDetector(mock_model, mock_config)
        
        assert detector.detect_model == mock_model
        assert detector.config == mock_config
        assert detector.bbox_conf_threshold == 0.6
        assert detector.class_id_to_label == {0: 'O', 1: 'X'}
        assert detector.class_id_to_player == {0: 1, 1: 2}

    def test_initialization_without_config(self, mock_model):
        """Test SymbolDetector initialization without configuration."""
        detector = SymbolDetector(mock_model)
        
        assert detector.detect_model == mock_model
        assert detector.config is None
        assert detector.bbox_conf_threshold == 0.5  # Default value
        assert detector.class_id_to_label == {0: 'O', 1: 'X'}  # Default mapping
        assert detector.class_id_to_player == {0: 1, 1: 2}  # Default mapping

    def test_initialization_with_custom_logger(self, mock_model):
        """Test SymbolDetector initialization with custom logger."""
        custom_logger = Mock()
        detector = SymbolDetector(mock_model, logger=custom_logger)
        
        assert detector.logger == custom_logger

    def test_initialization_default_logger(self, mock_model):
        """Test SymbolDetector initialization with default logger."""
        with patch('app.main.symbol_detector.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            detector = SymbolDetector(mock_model)
            
            mock_get_logger.assert_called_once_with('app.main.symbol_detector')
            assert detector.logger == mock_logger


class TestSymbolDetection:
    """Test symbol detection functionality."""

    def test_detect_symbols_basic(self, mock_model, mock_config, sample_frame):
        """Test basic symbol detection."""
        detector = SymbolDetector(mock_model, mock_config)
        
        # Mock model results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[10, 10, 50, 50]])
        mock_result.boxes.conf = np.array([0.8])
        mock_result.boxes.cls = np.array([0])
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(sample_frame)
        
        assert isinstance(annotated_frame, np.ndarray)
        assert isinstance(detections, list)
        assert len(detections) == 1
        assert detections[0]['label'] == 'O'  # class_id 0 maps to 'O'
        assert detections[0]['confidence'] == 0.8

    def test_detect_symbols_multiple_detections(self, mock_model, mock_config, sample_frame):
        """Test detection with multiple symbols."""
        detector = SymbolDetector(mock_model, mock_config)
        
        # Mock model results with multiple detections
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        mock_result.boxes.conf = np.array([0.8, 0.9])
        mock_result.boxes.cls = np.array([0, 1])
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(sample_frame)
        
        assert len(detections) == 2
        assert detections[0]['label'] == 'O'
        assert detections[1]['label'] == 'X'
        assert detections[0]['confidence'] == 0.8
        assert detections[1]['confidence'] == 0.9

    def test_detect_symbols_confidence_filtering(self, mock_model, mock_config, sample_frame):
        """Test confidence threshold filtering."""
        detector = SymbolDetector(mock_model, mock_config)
        
        # Mock model results with varying confidence levels
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        mock_result.boxes.conf = np.array([0.5, 0.8])  # One below threshold (0.6), one above
        mock_result.boxes.cls = np.array([0, 1])
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(sample_frame)
        
        # Only the detection with confidence >= 0.6 should be included
        assert len(detections) == 1
        assert detections[0]['confidence'] == 0.8
        assert detections[0]['label'] == 'X'

    def test_detect_symbols_no_detections(self, mock_model, mock_config, sample_frame):
        """Test when no symbols are detected."""
        detector = SymbolDetector(mock_model, mock_config)
        
        # Mock model results with no detections
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([]).reshape(0, 4)
        mock_result.boxes.conf = np.array([])
        mock_result.boxes.cls = np.array([])
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(sample_frame)
        
        assert isinstance(annotated_frame, np.ndarray)
        assert isinstance(detections, list)
        assert len(detections) == 0

    def test_detect_symbols_model_exception(self, mock_model, mock_config, sample_frame):
        """Test handling of model exceptions."""
        detector = SymbolDetector(mock_model, mock_config)
        
        # Mock model to raise exception
        mock_model.side_effect = Exception("Model inference failed")
        
        with patch.object(detector.logger, 'error') as mock_error:
            annotated_frame, detections = detector.detect_symbols(sample_frame)
            
            mock_error.assert_called_once()
            assert isinstance(annotated_frame, np.ndarray)
            assert len(detections) == 0


class TestSymbolMapping:
    """Test symbol mapping and label conversion."""

    def test_class_id_to_label_mapping(self, mock_model):
        """Test class ID to label mapping."""
        custom_config = Mock()
        custom_config.bbox_conf_threshold = 0.5
        custom_config.class_id_to_label = {0: 'CircleSymbol', 1: 'CrossSymbol'}
        custom_config.class_id_to_player = {0: 1, 1: 2}
        
        detector = SymbolDetector(mock_model, custom_config)
        assert detector.class_id_to_label[0] == 'CircleSymbol'
        assert detector.class_id_to_label[1] == 'CrossSymbol'

    def test_class_id_to_player_mapping(self, mock_model):
        """Test class ID to player mapping."""
        custom_config = Mock()
        custom_config.bbox_conf_threshold = 0.5
        custom_config.class_id_to_label = {0: 'O', 1: 'X'}
        custom_config.class_id_to_player = {0: 'Player1', 1: 'Player2'}
        
        detector = SymbolDetector(mock_model, custom_config)
        assert detector.class_id_to_player[0] == 'Player1'
        assert detector.class_id_to_player[1] == 'Player2'

    def test_inverted_label_mapping_fix(self, mock_model):
        """Test the inverted label mapping fix."""
        detector = SymbolDetector(mock_model)
        
        # The fix: class 0 should map to 'O', class 1 should map to 'X'
        assert detector.class_id_to_label[0] == 'O'
        assert detector.class_id_to_label[1] == 'X'
        
        # Player mapping should also be inverted accordingly
        assert detector.class_id_to_player[0] == 1  # O = player 1
        assert detector.class_id_to_player[1] == 2  # X = player 2


class TestDetectionBoundingBoxes:
    """Test bounding box processing and formatting."""

    def test_bounding_box_format(self, mock_model, mock_config, sample_frame):
        """Test that bounding boxes are properly formatted."""
        detector = SymbolDetector(mock_model, mock_config)
        
        # Mock model results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[10.5, 20.3, 50.7, 60.9]])
        mock_result.boxes.conf = np.array([0.8])
        mock_result.boxes.cls = np.array([0])
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(sample_frame)
        
        assert len(detections) == 1
        detection = detections[0]
        
        # Check that bounding box coordinates are properly converted
        assert 'box' in detection
        box = detection['box']
        assert len(box) == 4
        assert all(isinstance(coord, (int, float)) for coord in box)

    def test_detection_data_structure(self, mock_model, mock_config, sample_frame):
        """Test the structure of detection results."""
        detector = SymbolDetector(mock_model, mock_config)
        
        # Mock model results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[10, 20, 50, 60]])
        mock_result.boxes.conf = np.array([0.8])
        mock_result.boxes.cls = np.array([1])
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(sample_frame)
        
        assert len(detections) == 1
        detection = detections[0]
        
        # Check required fields
        required_fields = ['box', 'label', 'confidence', 'class_id']
        for field in required_fields:
            assert field in detection
        
        assert detection['label'] == 'X'
        assert detection['confidence'] == 0.8
        assert detection['class_id'] == 1


class TestSymbolDetectorConstants:
    """Test SymbolDetector constants and validation."""

    def test_has_required_attributes(self, mock_model, mock_config):
        """Test that SymbolDetector has all required attributes."""
        detector = SymbolDetector(mock_model, mock_config)
        
        required_attrs = [
            'detect_model', 'config', 'logger', 'bbox_conf_threshold',
            'class_id_to_label', 'class_id_to_player'
        ]
        for attr in required_attrs:
            assert hasattr(detector, attr), f"Missing attribute: {attr}"

    def test_has_required_methods(self, mock_model, mock_config):
        """Test that SymbolDetector has all required methods."""
        detector = SymbolDetector(mock_model, mock_config)
        
        required_methods = ['detect_symbols']
        for method in required_methods:
            assert hasattr(detector, method), f"Missing method: {method}"
            assert callable(getattr(detector, method)), f"Method {method} not callable"

    def test_confidence_threshold_validation(self, mock_model):
        """Test confidence threshold validation."""
        # Test with various threshold values
        thresholds = [0.0, 0.5, 0.9, 1.0]
        
        for threshold in thresholds:
            config = Mock()
            config.bbox_conf_threshold = threshold
            config.class_id_to_label = {0: 'O', 1: 'X'}
            config.class_id_to_player = {0: 1, 1: 2}
            
            detector = SymbolDetector(mock_model, config)
            assert detector.bbox_conf_threshold == threshold
            assert 0.0 <= detector.bbox_conf_threshold <= 1.0


class TestSymbolDetectorIntegration:
    """Test integration scenarios."""

    def test_full_detection_pipeline(self, mock_model, sample_frame):
        """Test complete detection pipeline."""
        # Create detector with default settings
        detector = SymbolDetector(mock_model)
        
        # Mock realistic model output
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[100, 100, 150, 150], [200, 200, 250, 250]])
        mock_result.boxes.conf = np.array([0.95, 0.87])
        mock_result.boxes.cls = np.array([0, 1])  # O and X
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(sample_frame)
        
        # Verify complete pipeline
        assert annotated_frame is not None
        assert len(detections) == 2
        
        # Check first detection (O)
        assert detections[0]['label'] == 'O'
        assert detections[0]['confidence'] == 0.95
        assert detections[0]['class_id'] == 0
        
        # Check second detection (X)
        assert detections[1]['label'] == 'X'
        assert detections[1]['confidence'] == 0.87
        assert detections[1]['class_id'] == 1

    def test_edge_case_empty_frame(self, mock_model):
        """Test detection with edge case inputs."""
        detector = SymbolDetector(mock_model)
        
        # Test with minimal frame
        empty_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Mock no detections
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([]).reshape(0, 4)
        mock_result.boxes.conf = np.array([])
        mock_result.boxes.cls = np.array([])
        mock_model.return_value = [mock_result]
        
        annotated_frame, detections = detector.detect_symbols(empty_frame)
        
        assert annotated_frame is not None
        assert len(detections) == 0