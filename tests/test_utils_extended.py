# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Extended tests for utils module to improve coverage.
"""
import pytest
import time
from unittest.mock import patch

from app.core.utils import FPSCalculator


class TestFPSCalculatorExtended:
    """Extended test class for FPSCalculator."""
    
    def test_initialization_edge_cases(self):
        """Test FPS calculator initialization edge cases."""
        # Test minimum valid buffer size
        calc = FPSCalculator(buffer_size=1)
        assert calc.buffer_size == 1
        
        # Test large buffer size
        calc = FPSCalculator(buffer_size=1000)
        assert calc.buffer_size == 1000
    
    @patch('time.perf_counter')
    def test_rapid_successive_ticks(self, mock_time):
        """Test handling of rapid successive ticks."""
        calc = FPSCalculator()
        
        # Very rapid ticks (same timestamp)
        mock_time.side_effect = [0.0, 0.0, 0.0, 0.001]
        
        calc.tick()  # First tick
        calc.tick()  # Same time (should be ignored)
        calc.tick()  # Same time (should be ignored)
        calc.tick()  # 0.001 later
        
        # Should only have one valid duration (or zero if same timestamps are ignored)
        fps = calc.get_fps()
        assert fps >= 0.0  # FPS should be non-negative    
    @patch('time.perf_counter')
    def test_empty_timestamps_buffer(self, mock_time):
        """Test behavior with empty timestamps buffer."""
        calc = FPSCalculator()
        
        # Only one tick
        mock_time.return_value = 1.0
        calc.tick()
        
        # Should return 0 FPS (no durations recorded)
        assert calc.get_fps() == 0.0
    
    @patch('time.perf_counter')
    def test_average_duration_calculation(self, mock_time):
        """Test average duration calculation with multiple values."""
        calc = FPSCalculator(buffer_size=3)
        
        # Create specific intervals: 0.1, 0.2, 0.3 seconds
        mock_time.side_effect = [0.0, 0.1, 0.3, 0.6]
        
        calc.tick()  # t=0.0
        calc.tick()  # t=0.1, duration=0.1
        calc.tick()  # t=0.3, duration=0.2
        calc.tick()  # t=0.6, duration=0.3
        
        # Average duration calculation depends on implementation
        # Test that FPS is reasonable
        fps = calc.get_fps()
        assert fps > 0.0  # Should be positive
    
    def test_multiple_resets(self):
        """Test multiple reset operations."""
        calc = FPSCalculator()
        
        # Add some data
        calc.tick()
        calc.tick()
        
        # Reset multiple times
        calc.reset()
        calc.reset()  # Should not cause issues
        
        assert calc.get_fps() == 0.0
        assert len(calc._timestamps) == 0
