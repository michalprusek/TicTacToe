"""
Unit tests for SymbolDetector class.
"""
import pytest
from unittest.mock import MagicMock
import numpy as np

from app.main.symbol_detector import SymbolDetector


@pytest.fixture
def mock_detect_model():
    """mock_detect_model fixture for tests."""
    mock_model = MagicMock()
    
    # Mock predict method to return a result with boxes
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    # Create test data for 2 detections (X and O)
    boxes_xyxy = np.array([
        [100, 100, 150, 150],  # X symbol
        [200, 200, 250, 250]   # O symbol
    ])
    confs = np.array([0.8, 0.9])
    cls_ids = np.array([0, 1])  # 0=X, 1=O
    
    # Set up the mock chain
    mock_boxes.xyxy = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = boxes_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value.numpy.return_value = confs
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value.numpy.return_value = cls_ids
    
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]
    
    return mock_model


@pytest.fixture
def mock_config():
    """mock_config fixture for tests."""
    mock_config = MagicMock()
    mock_config.bbox_conf_threshold = 0.5
    mock_config.class_id_to_label = {0: 'X', 1: 'O'}
    mock_config.class_id_to_player = {0: 1, 1: 2}  # X=1, O=2
    return mock_config


@pytest.fixture
def symbol_detector(mock_detect_model, mock_config):
    """symbol_detector fixture for tests."""
    detector = SymbolDetector(
        detect_model=mock_detect_model,
        config=mock_config
    )
    return detector


class TestSymbolDetector:
    """Test SymbolDetector class."""

    def test_init(self, symbol_detector, mock_detect_model, mock_config):
        """Test initialization."""
        assert symbol_detector.detect_model == mock_detect_model
        assert symbol_detector.config == mock_config
        assert symbol_detector.bbox_conf_threshold == mock_config.bbox_conf_threshold
        assert symbol_detector.class_id_to_label == mock_config.class_id_to_label
        assert symbol_detector.class_id_to_player == mock_config.class_id_to_player

    def test_detect_symbols(self, symbol_detector, mock_detect_model):
        """Test detect_symbols method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Call the method
        _, symbols = symbol_detector.detect_symbols(frame)
        
        # Check that detect_model.predict was called
        mock_detect_model.predict.assert_called_once()
        
        # Check that symbols were extracted
        assert len(symbols) == 2
        
        # Check first symbol (X)
        assert symbols[0]['label'] == 'X'
        assert symbols[0]['confidence'] == 0.8
        assert symbols[0]['box'] == [100, 100, 150, 150]
        assert symbols[0]['class_id'] == 0
        
        # Check second symbol (O)
        assert symbols[1]['label'] == 'O'
        assert symbols[1]['confidence'] == 0.9
        assert symbols[1]['box'] == [200, 200, 250, 250]
        assert symbols[1]['class_id'] == 1

    def test_detect_symbols_no_results(self, symbol_detector, mock_detect_model):
        """Test detect_symbols method with no results."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set up mock to return empty results
        mock_detect_model.predict.return_value = []
        
        # Call the method
        _, symbols = symbol_detector.detect_symbols(frame)
        
        # Check that detect_model.predict was called
        mock_detect_model.predict.assert_called_once()
        
        # Check that symbols list is empty
        assert len(symbols) == 0

    def test_get_nearest_cell_inside(self, symbol_detector):
        """Test get_nearest_cell method with point inside a cell."""
        # Create test cell polygons (3x3 grid)
        cell_polygons = []
        for i in range(3):
            for j in range(3):
                # Create a square cell
                cell = np.array([
                    [j*100, i*100],
                    [(j+1)*100, i*100],
                    [(j+1)*100, (i+1)*100],
                    [j*100, (i+1)*100]
                ])
                cell_polygons.append(cell)
        
        # Test point inside cell (1, 1) - center cell
        x, y = 150, 150
        cell_coords = symbol_detector.get_nearest_cell(x, y, cell_polygons)
        
        # Should return row=1, col=1
        assert cell_coords == (1, 1)

    def test_get_nearest_cell_outside(self, symbol_detector):
        """Test get_nearest_cell method with point outside all cells."""
        # Create test cell polygons (3x3 grid)
        cell_polygons = []
        for i in range(3):
            for j in range(3):
                # Create a square cell
                cell = np.array([
                    [j*100, i*100],
                    [(j+1)*100, i*100],
                    [(j+1)*100, (i+1)*100],
                    [j*100, (i+1)*100]
                ])
                cell_polygons.append(cell)
        
        # Test point outside all cells but closest to cell (0, 0)
        x, y = -10, -10
        cell_coords = symbol_detector.get_nearest_cell(x, y, cell_polygons)
        
        # Should return row=0, col=0
        assert cell_coords == (0, 0)

    def test_get_nearest_cell_no_polygons(self, symbol_detector):
        """Test get_nearest_cell method with no cell polygons."""
        # Call with empty cell_polygons
        cell_coords = symbol_detector.get_nearest_cell(100, 100, [])
        
        # Should return None
        assert cell_coords is None

    def test_assign_symbols_to_cells(self, symbol_detector):
        """Test assign_symbols_to_cells method."""
        # Create test symbols
        symbols = [
            {'label': 'X', 'confidence': 0.8, 'box': [110, 110, 140, 140], 'class_id': 0},
            {'label': 'O', 'confidence': 0.9, 'box': [210, 210, 240, 240], 'class_id': 1}
        ]
        
        # Create test cell polygons (3x3 grid)
        cell_polygons = []
        for i in range(3):
            for j in range(3):
                # Create a square cell
                cell = np.array([
                    [j*100, i*100],
                    [(j+1)*100, i*100],
                    [(j+1)*100, (i+1)*100],
                    [j*100, (i+1)*100]
                ])
                cell_polygons.append(cell)
        
        # Call the method
        assigned_symbols = symbol_detector.assign_symbols_to_cells(symbols, cell_polygons)
        
        # Should assign X to cell (1, 1) and O to cell (2, 2)
        assert len(assigned_symbols) == 2
        assert assigned_symbols[0] == (1, 1, 'X')
        assert assigned_symbols[1] == (2, 2, 'O')

    def test_assign_symbols_to_cells_empty(self, symbol_detector):
        """Test assign_symbols_to_cells method with empty inputs."""
        # Call with empty symbols
        assigned_symbols = symbol_detector.assign_symbols_to_cells([], [])
        
        # Should return empty list
        assert assigned_symbols == []
