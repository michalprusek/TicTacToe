# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and symbol caching implementation
"""
Test symbol caching system for handling occlusion during gameplay.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock

from app.core.symbol_cache import SymbolCache, CachedSymbol, OcclusionDetector
from app.core.game_state import EMPTY


class TestCachedSymbol:
    """Test CachedSymbol class."""

    def test_cached_symbol_creation(self):
        """Test creating a cached symbol."""
        center_uv = np.array([100, 100])
        symbol = CachedSymbol('X', (0, 0), 0.95, time.time(), center_uv)
        
        assert symbol.symbol == 'X'
        assert symbol.position == (0, 0)
        assert symbol.confidence == 0.95
        assert symbol.occlusion_count == 0
        assert symbol.total_detections == 1
        assert not symbol.is_persistent
        assert np.array_equal(symbol.center_uv, center_uv)


class TestOcclusionDetector:
    """Test OcclusionDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = OcclusionDetector()
        self.cell_centers = np.array([
            [100, 100], [200, 100], [300, 100],  # Row 0
            [100, 200], [200, 200], [300, 200],  # Row 1
            [100, 300], [200, 300], [300, 300]   # Row 2
        ], dtype=np.float32)

    def test_no_occlusion_when_all_symbols_detected(self):
        """Test that no occlusion is detected when all symbols are visible."""
        # Create cached symbols
        cached_symbols = {
            (0, 0): CachedSymbol('X', (0, 0), 0.95, time.time(), np.array([100, 100])),
            (1, 1): CachedSymbol('O', (1, 1), 0.92, time.time(), np.array([200, 200]))
        }
        
        # Create current detections matching cached symbols
        current_detections = [
            {'center_uv': np.array([100, 100]), 'confidence': 0.95},
            {'center_uv': np.array([200, 200]), 'confidence': 0.92}
        ]
        
        occluded = self.detector.detect_occlusion(
            current_detections, cached_symbols, self.cell_centers
        )
        
        assert len(occluded) == 0

    def test_occlusion_detected_when_symbols_missing(self):
        """Test that occlusion is detected when cached symbols are missing."""
        # Create cached symbols
        cached_symbols = {
            (0, 0): CachedSymbol('X', (0, 0), 0.95, time.time(), np.array([100, 100])),
            (1, 1): CachedSymbol('O', (1, 1), 0.92, time.time(), np.array([200, 200]))
        }
        
        # Current detections missing one symbol (simulating occlusion)
        current_detections = [
            {'center_uv': np.array([100, 100]), 'confidence': 0.95}
        ]
        
        # Build detection history to trigger occlusion detection
        self.detector.detection_history = [2, 2, 2, 1]  # Sudden drop
        
        occluded = self.detector.detect_occlusion(
            current_detections, cached_symbols, self.cell_centers
        )
        
        assert (1, 1) in occluded

    def test_find_closest_cell(self):
        """Test finding closest cell to UV coordinates."""
        # Test center of cell (0,0)
        pos = self.detector._find_closest_cell(
            np.array([100, 100]), self.cell_centers
        )
        assert pos == (0, 0)
        
        # Test center of cell (1,1)
        pos = self.detector._find_closest_cell(
            np.array([200, 200]), self.cell_centers
        )
        assert pos == (1, 1)
        
        # Test center of cell (2,2)
        pos = self.detector._find_closest_cell(
            np.array([300, 300]), self.cell_centers
        )
        assert pos == (2, 2)


class TestSymbolCache:
    """Test SymbolCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = SymbolCache()
        self.cache.logger = Mock()
        
        self.cell_centers = np.array([
            [100, 100], [200, 100], [300, 100],  # Row 0
            [100, 200], [200, 200], [300, 200],  # Row 1
            [100, 300], [200, 300], [300, 300]   # Row 2
        ], dtype=np.float32)
        
        self.class_id_to_player = {0: 'X', 1: 'O'}

    def test_cache_high_confidence_symbols(self):
        """Test that high confidence symbols are cached."""
        detections = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [95, 95, 105, 105],  # Center at (100, 100)
                'class_id': 0
            }
        ]
        
        board, using_cached = self.cache.update_cache(
            detections, self.cell_centers, time.time(), self.class_id_to_player
        )
        
        assert board[0][0] == 'X'
        assert (0, 0) in self.cache.cached_symbols
        assert not using_cached  # First detection, not using cache yet

    def test_reject_low_confidence_symbols(self):
        """Test that low confidence symbols are rejected."""
        detections = [
            {
                'label': 'X',
                'confidence': 0.5,  # Below threshold
                'box': [95, 95, 105, 105],
                'class_id': 0
            }
        ]
        
        board, using_cached = self.cache.update_cache(
            detections, self.cell_centers, time.time(), self.class_id_to_player
        )
        
        # Board should be empty
        expected_board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
        assert board == expected_board
        assert len(self.cache.cached_symbols) == 0

    def test_symbol_persistence_during_occlusion(self):
        """Test that symbols persist during temporary occlusion."""
        timestamp = time.time()
        
        # First, cache a symbol
        detections = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [95, 95, 105, 105],
                'class_id': 0
            }
        ]
        
        board1, _ = self.cache.update_cache(
            detections, self.cell_centers, timestamp, self.class_id_to_player
        )
        assert board1[0][0] == 'X'
        
        # Now simulate occlusion (no detections)
        board2, using_cached = self.cache.update_cache(
            [], self.cell_centers, timestamp + 1, self.class_id_to_player
        )
        
        # Symbol should still be there due to caching
        assert board2[0][0] == 'X'
        assert using_cached  # Should be using cached data

    def test_cache_cleanup_after_timeout(self):
        """Test that old cached symbols are cleaned up."""
        timestamp = time.time()
        
        # Cache a symbol
        detections = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [95, 95, 105, 105],
                'class_id': 0
            }
        ]
        
        self.cache.update_cache(
            detections, self.cell_centers, timestamp, self.class_id_to_player
        )
        
        # Simulate time passing beyond cleanup threshold
        old_timestamp = timestamp + self.cache.max_occlusion_time + 1
        
        board, _ = self.cache.update_cache(
            [], self.cell_centers, old_timestamp, self.class_id_to_player
        )
        
        # Symbol should be cleaned up
        expected_board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
        assert board == expected_board
        assert len(self.cache.cached_symbols) == 0

    def test_cache_clear(self):
        """Test clearing the cache."""
        # Add some symbols to cache
        detections = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [95, 95, 105, 105],
                'class_id': 0
            }
        ]
        
        self.cache.update_cache(
            detections, self.cell_centers, time.time(), self.class_id_to_player
        )
        
        assert len(self.cache.cached_symbols) > 0
        
        # Clear cache
        self.cache.clear_cache()
        
        assert len(self.cache.cached_symbols) == 0
        assert not self.cache.using_cached_data

    def test_cache_status(self):
        """Test getting cache status."""
        status = self.cache.get_cache_status()
        
        assert 'cached_count' in status
        assert 'using_cached_data' in status
        assert 'cached_positions' in status
        assert 'last_update' in status
        
        assert status['cached_count'] == 0
        assert not status['using_cached_data']
        assert status['cached_positions'] == []

    def test_symbol_update_resets_occlusion_count(self):
        """Test that detecting a cached symbol resets its occlusion count."""
        timestamp = time.time()
        
        # Cache a symbol
        detections = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [95, 95, 105, 105],
                'class_id': 0
            }
        ]
        
        self.cache.update_cache(
            detections, self.cell_centers, timestamp, self.class_id_to_player
        )
        
        # Simulate occlusion (increase occlusion count)
        self.cache.update_cache(
            [], self.cell_centers, timestamp + 1, self.class_id_to_player
        )
        
        cached_symbol = self.cache.cached_symbols[(0, 0)]
        assert cached_symbol.occlusion_count > 0
        
        # Detect symbol again
        self.cache.update_cache(
            detections, self.cell_centers, timestamp + 2, self.class_id_to_player
        )
        
        # Occlusion count should be reset
        assert cached_symbol.occlusion_count == 0
        assert cached_symbol.total_detections == 2
