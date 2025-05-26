"""
Comprehensive tests for app.core.utils module using pytest.
Tests FPSCalculator class with all scenarios including edge cases.
"""

import pytest
import time
from unittest.mock import patch
from app.core.utils import FPSCalculator


class TestFPSCalculator:
    """Test FPSCalculator class functionality."""
    
    def test_init_default_buffer_size(self):
        """Test FPSCalculator initialization with default buffer size."""
        calc = FPSCalculator()
        assert calc.buffer_size == 10
        assert len(calc._timestamps) == 0
        assert calc._last_tick_time == 0.0
    
    def test_init_custom_buffer_size(self):
        """Test FPSCalculator initialization with custom buffer size."""
        calc = FPSCalculator(buffer_size=5)
        assert calc.buffer_size == 5
        assert calc._timestamps.maxlen == 5
    
    def test_init_invalid_buffer_size(self):
        """Test FPSCalculator initialization with invalid buffer size."""
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=0)
        
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=-1)
    
    def test_tick_first_call(self):
        """Test first tick call doesn't add timestamp."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', return_value=1.0):
            calc.tick()
        
        assert len(calc._timestamps) == 0
        assert calc._last_tick_time == 1.0
    
    def test_tick_second_call(self):
        """Test second tick call adds duration."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1]):
            calc.tick()  # First tick
            calc.tick()  # Second tick
        
        assert len(calc._timestamps) == 1
        assert calc._timestamps[0] == pytest.approx(0.1)
        assert calc._last_tick_time == 1.1
    
    def test_tick_multiple_calls(self):
        """Test multiple tick calls."""
        calc = FPSCalculator(buffer_size=3)
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.2, 1.3, 1.4]):
            calc.tick()  # t=1.0
            calc.tick()  # t=1.1, duration=0.1
            calc.tick()  # t=1.2, duration=0.1  
            calc.tick()  # t=1.3, duration=0.1
            calc.tick()  # t=1.4, duration=0.1
        
        # Should only keep last 3 durations due to buffer_size=3
        assert len(calc._timestamps) == 3
        assert all(abs(d - 0.1) < 1e-10 for d in calc._timestamps)
    
    def test_tick_zero_duration(self):
        """Test tick with zero duration is not added."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[1.0, 1.0]):
            calc.tick()  # First tick
            calc.tick()  # Same time, duration=0
        
        assert len(calc._timestamps) == 0  # Zero duration not added
    
    def test_get_fps_no_timestamps(self):
        """Test get_fps with no timestamps returns 0.0."""
        calc = FPSCalculator()
        assert calc.get_fps() == 0.0
    
    def test_get_fps_single_timestamp(self):
        """Test get_fps with single timestamp."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1]):
            calc.tick()
            calc.tick()
        
        fps = calc.get_fps()
        assert fps == pytest.approx(10.0)  # 1 / 0.1 = 10 FPS
    
    def test_get_fps_multiple_timestamps(self):
        """Test get_fps with multiple timestamps."""
        calc = FPSCalculator()
        
        # Simulate 20 FPS (0.05 second intervals)
        with patch('time.perf_counter', side_effect=[1.0, 1.05, 1.10, 1.15]):
            calc.tick()
            calc.tick()  # duration=0.05
            calc.tick()  # duration=0.05
            calc.tick()  # duration=0.05
        
        fps = calc.get_fps()
        assert fps == pytest.approx(20.0)  # 1 / 0.05 = 20 FPS
    
    def test_get_fps_varying_durations(self):
        """Test get_fps with varying durations."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.3]):
            calc.tick()
            calc.tick()  # duration=0.1
            calc.tick()  # duration=0.2
        
        # Average duration = (0.1 + 0.2) / 2 = 0.15
        # FPS = 1 / 0.15 = 6.67
        fps = calc.get_fps()
        assert fps == pytest.approx(6.666666666666667)
    
    def test_reset(self):
        """Test reset clears all data."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.2]):
            calc.tick()
            calc.tick()
            calc.tick()
        
        assert len(calc._timestamps) == 2
        assert calc._last_tick_time == 1.2
        
        calc.reset()
        
        assert len(calc._timestamps) == 0
        assert calc._last_tick_time == 0.0
        assert calc.get_fps() == 0.0
    
    def test_buffer_overflow(self):
        """Test buffer respects maxlen and removes old entries."""
        calc = FPSCalculator(buffer_size=2)
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.2, 1.3, 1.4]):
            calc.tick()  # t=1.0
            calc.tick()  # t=1.1, duration=0.1
            calc.tick()  # t=1.2, duration=0.1
            calc.tick()  # t=1.3, duration=0.1 (should evict first 0.1)
            calc.tick()  # t=1.4, duration=0.1 (should evict second 0.1)
        
        assert len(calc._timestamps) == 2
        assert all(abs(d - 0.1) < 1e-10 for d in calc._timestamps)
    
    def test_real_time_integration(self):
        """Test with real time.perf_counter (integration test)."""
        calc = FPSCalculator(buffer_size=5)
        
        calc.tick()
        time.sleep(0.01)  # 10ms
        calc.tick()
        time.sleep(0.01)  # 10ms  
        calc.tick()
        
        fps = calc.get_fps()
        # Should be around 100 FPS (1/0.01), allow some tolerance
        assert 50 < fps < 200
    
    def test_edge_case_very_small_duration(self):
        """Test with very small but non-zero duration."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[1.0, 1.000001]):
            calc.tick()
            calc.tick()
        
        fps = calc.get_fps()
        assert fps == pytest.approx(1000000.0)  # 1 / 0.000001
    
    def test_edge_case_negative_time_jump(self):
        """Test handling of negative time jump (should not happen but test robustness)."""
        calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[2.0, 1.0]):
            calc.tick()
            calc.tick()  # Negative duration
        
        # Negative duration should not be added
        assert len(calc._timestamps) == 0
        assert calc.get_fps() == 0.0