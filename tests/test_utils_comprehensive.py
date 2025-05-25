"""
Comprehensive pytest test suite for core/utils.py module.
Tests FPSCalculator functionality.
"""

import pytest
import time
from unittest.mock import patch, Mock
from app.core.utils import FPSCalculator


class TestFPSCalculator:
    """Test FPSCalculator class functionality."""
    
    def test_fps_calculator_initialization_default(self):
        """Test FPSCalculator initialization with default buffer size."""
        fps_calc = FPSCalculator()
        assert fps_calc.buffer_size == 10
        assert len(fps_calc._timestamps) == 0
        assert fps_calc._last_tick_time == 0.0
    
    def test_fps_calculator_initialization_custom_buffer(self):
        """Test FPSCalculator initialization with custom buffer size."""
        fps_calc = FPSCalculator(buffer_size=5)
        assert fps_calc.buffer_size == 5
        assert len(fps_calc._timestamps) == 0
        assert fps_calc._last_tick_time == 0.0
    
    def test_fps_calculator_initialization_invalid_buffer_size(self):
        """Test FPSCalculator initialization with invalid buffer size."""
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=0)
        
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=-5)
    
    def test_fps_calculator_first_tick(self):
        """Test first tick behavior."""
        fps_calc = FPSCalculator()
        
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
        
        # First tick should set last_tick_time but not add to timestamps
        assert fps_calc._last_tick_time == 1.0
        assert len(fps_calc._timestamps) == 0
    
    def test_fps_calculator_second_tick(self):
        """Test second tick adds duration to timestamps."""
        fps_calc = FPSCalculator()
        
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
        
        with patch('time.perf_counter', return_value=1.1):
            fps_calc.tick()
        
        # Second tick should add duration (0.1 seconds)
        assert fps_calc._last_tick_time == 1.1
        assert len(fps_calc._timestamps) == 1
        assert fps_calc._timestamps[0] == 0.1
    
    def test_fps_calculator_multiple_ticks(self):
        """Test multiple ticks behavior."""
        fps_calc = FPSCalculator(buffer_size=3)
        
        mock_times = [1.0, 1.1, 1.3, 1.4, 1.7]
        expected_durations = [0.1, 0.2, 0.1, 0.3]  # differences between consecutive times
        
        for i, mock_time in enumerate(mock_times):
            with patch('time.perf_counter', return_value=mock_time):
                fps_calc.tick()
        
        # Should only keep last 3 durations due to buffer size
        assert len(fps_calc._timestamps) == 3
        assert list(fps_calc._timestamps) == expected_durations[-3:]  # [0.2, 0.1, 0.3]
    
    def test_fps_calculator_get_fps_no_data(self):
        """Test get_fps returns 0.0 when no timestamps available."""
        fps_calc = FPSCalculator()
        assert fps_calc.get_fps() == 0.0
    
    def test_fps_calculator_get_fps_single_duration(self):
        """Test get_fps with single duration."""
        fps_calc = FPSCalculator()
        
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
        
        with patch('time.perf_counter', return_value=1.1):  # 0.1 second duration
            fps_calc.tick()
        
        fps = fps_calc.get_fps()
        assert fps == 10.0  # 1 / 0.1 = 10 FPS
    
    def test_fps_calculator_get_fps_multiple_durations(self):
        """Test get_fps with multiple durations calculates average."""
        fps_calc = FPSCalculator()
        
        # Add durations: 0.1, 0.2, 0.1 (average = 0.1333...)
        mock_times = [1.0, 1.1, 1.3, 1.4]
        
        for mock_time in mock_times:
            with patch('time.perf_counter', return_value=mock_time):
                fps_calc.tick()
        
        fps = fps_calc.get_fps()
        expected_avg_duration = (0.1 + 0.2 + 0.1) / 3  # 0.1333...
        expected_fps = 1.0 / expected_avg_duration  # ~7.5
        assert abs(fps - expected_fps) < 0.01
    
    def test_fps_calculator_get_fps_zero_duration(self):
        """Test get_fps handles zero duration gracefully."""
        fps_calc = FPSCalculator()
        
        # Simulate two ticks at exact same time
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
            fps_calc.tick()
        
        # Should return 0.0 for zero duration
        assert fps_calc.get_fps() == 0.0
    
    def test_fps_calculator_reset(self):
        """Test reset functionality."""
        fps_calc = FPSCalculator()
        
        # Add some data
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
        
        with patch('time.perf_counter', return_value=1.1):
            fps_calc.tick()
        
        # Verify data exists
        assert len(fps_calc._timestamps) == 1
        assert fps_calc._last_tick_time == 1.1
        
        # Reset
        fps_calc.reset()
        
        # Verify data is cleared
        assert len(fps_calc._timestamps) == 0
        assert fps_calc._last_tick_time == 0.0
        assert fps_calc.get_fps() == 0.0
    
    def test_fps_calculator_buffer_overflow(self):
        """Test buffer overflow behavior with deque maxlen."""
        fps_calc = FPSCalculator(buffer_size=2)
        
        mock_times = [1.0, 1.1, 1.3, 1.4, 1.7]
        
        for mock_time in mock_times:
            with patch('time.perf_counter', return_value=mock_time):
                fps_calc.tick()
        
        # Should only keep last 2 durations: [0.1, 0.3]
        assert len(fps_calc._timestamps) == 2
        assert list(fps_calc._timestamps) == [0.1, 0.3]
    
    def test_fps_calculator_very_small_durations(self):
        """Test FPS calculator with very small durations."""
        fps_calc = FPSCalculator()
        
        # Simulate 60 FPS (duration ~0.0167 seconds)
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
        
        with patch('time.perf_counter', return_value=1.0167):
            fps_calc.tick()
        
        fps = fps_calc.get_fps()
        assert 59 < fps < 61  # Should be approximately 60 FPS
    
    def test_fps_calculator_large_durations(self):
        """Test FPS calculator with large durations."""
        fps_calc = FPSCalculator()
        
        # Simulate 1 FPS (1 second duration)
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
        
        with patch('time.perf_counter', return_value=2.0):
            fps_calc.tick()
        
        fps = fps_calc.get_fps()
        assert fps == 1.0
    
    def test_fps_calculator_realistic_scenario(self):
        """Test FPS calculator with realistic timing scenario."""
        fps_calc = FPSCalculator(buffer_size=10)
        
        # Simulate irregular frame times
        base_time = 10.0
        frame_durations = [0.016, 0.017, 0.033, 0.016, 0.016, 0.050, 0.016, 0.017]
        
        current_time = base_time
        with patch('time.perf_counter', return_value=current_time):
            fps_calc.tick()
        
        for duration in frame_durations:
            current_time += duration
            with patch('time.perf_counter', return_value=current_time):
                fps_calc.tick()
        
        fps = fps_calc.get_fps()
        # Average duration should be around the mean of frame_durations
        avg_duration = sum(frame_durations) / len(frame_durations)
        expected_fps = 1.0 / avg_duration
        assert abs(fps - expected_fps) < 0.1
    
    def test_fps_calculator_edge_case_negative_duration(self):
        """Test edge case where time goes backwards (shouldn't happen but test anyway)."""
        fps_calc = FPSCalculator()
        
        with patch('time.perf_counter', return_value=2.0):
            fps_calc.tick()
        
        # Time goes backwards
        with patch('time.perf_counter', return_value=1.0):
            fps_calc.tick()
        
        # Should not add negative duration
        assert len(fps_calc._timestamps) == 0
        assert fps_calc.get_fps() == 0.0


class TestFPSCalculatorIntegration:
    """Integration tests for FPSCalculator."""
    
    def test_fps_calculator_real_time_simulation(self):
        """Test FPS calculator with real time simulation."""
        fps_calc = FPSCalculator(buffer_size=5)
        
        # Simulate actual time progression
        start_time = time.perf_counter()
        fps_calc.tick()
        
        # Add small delay and tick again
        time.sleep(0.01)  # 10ms delay
        fps_calc.tick()
        
        fps = fps_calc.get_fps()
        # Should be roughly 100 FPS (1 / 0.01)
        assert 80 < fps < 120  # Allow some variance due to timing precision
    
    def test_fps_calculator_consistency(self):
        """Test that FPS calculator provides consistent results."""
        fps_calc = FPSCalculator()
        
        # Add consistent durations
        mock_times = [i * 0.1 for i in range(10)]  # 0.0, 0.1, 0.2, ... 0.9
        
        for mock_time in mock_times:
            with patch('time.perf_counter', return_value=mock_time):
                fps_calc.tick()
        
        fps = fps_calc.get_fps()
        assert fps == 10.0  # 1 / 0.1 = 10 FPS


if __name__ == "__main__":
    pytest.main([__file__])