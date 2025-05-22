import time
from collections import deque

class FPSCalculator:
    """Calculates Frames Per Second (FPS) over a sliding window."""
    def __init__(self, buffer_size: int = 10):
        """Initializes the FPS calculator.

        Args:
            buffer_size: The number of recent frame timestamps to store for averaging.
        """
        if buffer_size <= 0:
            raise ValueError("Buffer size must be a positive integer.")
        self.buffer_size = buffer_size
        self._timestamps = deque(maxlen=buffer_size)
        self._last_tick_time: float = 0.0

    def tick(self) -> None:
        """Records a new frame timestamp (or event)."""
        current_time = time.perf_counter()
        if self._last_tick_time > 0: # Ensure there's a previous tick to calculate duration
            # Store duration between ticks for more accurate FPS
            duration = current_time - self._last_tick_time
            if duration > 0: # Avoid division by zero if ticks are too close
                self._timestamps.append(duration)
        self._last_tick_time = current_time

    def get_fps(self) -> float:
        """Calculates the current FPS based on the stored timestamps.

        Returns:
            The calculated FPS. Returns 0.0 if not enough data is available.
        """
        if not self._timestamps: # No durations recorded yet
            return 0.0
        
        # Calculate FPS based on the average duration of frames in the buffer
        avg_duration = sum(self._timestamps) / len(self._timestamps)
        if avg_duration > 0:
            return 1.0 / avg_duration
        return 0.0

    def reset(self) -> None:
        """Resets the calculator, clearing all stored timestamps."""
        self._timestamps.clear()
        self._last_tick_time = 0.0
