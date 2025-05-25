#!/usr/bin/env python3
"""
Test script for immediate difficulty synchronization between GUI and Bernoulli strategy selector.
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from app.core.strategy import BernoulliStrategySelector
from app.main.pyqt_gui import TicTacToeApp


class TestDifficultySynchronization(unittest.TestCase):
    """Test immediate difficulty synchronization between GUI and strategy selector"""

    @classmethod
    def setUpClass(cls):
        """Set up QApplication for all tests"""
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal TicTacToe app for testing
        self.gui_app = TicTacToeApp()

        # Mock the camera thread to avoid camera initialization
        self.gui_app.camera_thread = Mock()
        self.gui_app.camera_thread.start = Mock()

        # Ensure strategy selector is initialized
        if not hasattr(self.gui_app, 'strategy_selector') or not self.gui_app.strategy_selector:
            self.gui_app.strategy_selector = BernoulliStrategySelector(difficulty=5)

    def test_immediate_difficulty_sync(self):
        """Test that difficulty changes are immediately synchronized"""
        print("\n🎯 Testing immediate difficulty synchronization...")

        # Test different difficulty values
        test_values = [0, 3, 5, 7, 10]

        for difficulty in test_values:
            # Simulate GUI slider change
            self.gui_app.handle_difficulty_changed(difficulty)

            # Verify immediate synchronization
            actual_difficulty = self.gui_app.strategy_selector.difficulty
            actual_p = self.gui_app.strategy_selector.p
            expected_p = difficulty / 10.0

            self.assertEqual(actual_difficulty, difficulty,
                           f"Difficulty not synchronized: expected {difficulty}, got {actual_difficulty}")
            self.assertAlmostEqual(actual_p, expected_p, places=2,
                                 msg=f"Probability not synchronized: expected {expected_p}, got {actual_p}")

            print(f"✅ Difficulty {difficulty} → p={actual_p:.2f} (synchronized)")

    def test_real_time_updates(self):
        """Test rapid difficulty changes simulate real-time slider movement"""
        print("\n⚡ Testing real-time updates...")

        # Simulate rapid slider movements
        rapid_changes = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0]

        for i, difficulty in enumerate(rapid_changes):
            self.gui_app.handle_difficulty_changed(difficulty)

            # Verify each change is immediately applied
            actual_p = self.gui_app.strategy_selector.p
            expected_p = difficulty / 10.0

            self.assertAlmostEqual(actual_p, expected_p, places=2,
                                 msg=f"Rapid change {i}: expected p={expected_p}, got p={actual_p}")

            print(f"⚡ Rapid change {i}: difficulty={difficulty} → p={actual_p:.2f}")

    def test_strategy_behavior_changes(self):
        """Test that strategy behavior actually changes with difficulty"""
        print("\n🎲 Testing strategy behavior changes...")

        # Test extreme values
        test_cases = [
            (0, "random", 0.0),
            (5, "mixed", 0.5),
            (10, "intelligent", 1.0)
        ]

        for difficulty, expected_behavior, expected_p in test_cases:
            self.gui_app.handle_difficulty_changed(difficulty)

            # Verify probability
            actual_p = self.gui_app.strategy_selector.p
            self.assertAlmostEqual(actual_p, expected_p, places=2)

            # Test strategy selection behavior
            strategy_counts = {"random": 0, "minimax": 0}
            num_tests = 100

            for _ in range(num_tests):
                strategy = self.gui_app.strategy_selector.select_strategy()
                strategy_name = strategy.__class__.__name__.lower()
                if "random" in strategy_name:
                    strategy_counts["random"] += 1
                else:
                    strategy_counts["minimax"] += 1

            minimax_ratio = strategy_counts["minimax"] / num_tests

            print(f"🎲 Difficulty {difficulty} ({expected_behavior}): "
                  f"Minimax ratio = {minimax_ratio:.2f} (expected ≈{expected_p:.2f})")

            # Allow some tolerance for randomness
            if expected_p == 0.0:
                self.assertLess(minimax_ratio, 0.1, "Should be mostly random")
            elif expected_p == 1.0:
                self.assertGreater(minimax_ratio, 0.9, "Should be mostly intelligent")
            else:
                self.assertGreater(minimax_ratio, 0.3, "Should have some intelligent moves")
                self.assertLess(minimax_ratio, 0.7, "Should have some random moves")

    def test_gui_label_update(self):
        """Test that GUI difficulty label is updated immediately"""
        print("\n🏷️ Testing GUI label updates...")

        # Mock the difficulty label
        self.gui_app.difficulty_value_label = Mock()

        test_values = [0, 5, 10]
        for difficulty in test_values:
            self.gui_app.handle_difficulty_changed(difficulty)

            # Verify label was updated
            self.gui_app.difficulty_value_label.setText.assert_called_with(str(difficulty))
            print(f"🏷️ Label updated for difficulty {difficulty}")

    def test_strategy_description(self):
        """Test strategy description helper method"""
        print("\n📝 Testing strategy descriptions...")

        test_cases = [
            (0.0, "Náhodné tahy (0% inteligence)"),
            (0.5, "Smíšené tahy (50% inteligence)"),
            (1.0, "Inteligentní tahy (100% inteligence)"),
            (0.3, "Smíšené tahy (30% inteligence)"),
            (0.7, "Smíšené tahy (70% inteligence)")
        ]

        for p_value, expected_description in test_cases:
            actual_description = self.gui_app._get_strategy_description(p_value)
            self.assertEqual(actual_description, expected_description)
            print(f"📝 p={p_value:.1f} → '{actual_description}'")

    def test_no_delay_synchronization(self):
        """Test that synchronization happens without any delay"""
        print("\n⏱️ Testing no-delay synchronization...")

        # Record time before and after difficulty change
        start_time = time.time()
        self.gui_app.handle_difficulty_changed(7)
        end_time = time.time()

        # Verify synchronization happened immediately
        actual_p = self.gui_app.strategy_selector.p
        expected_p = 0.7

        self.assertAlmostEqual(actual_p, expected_p, places=2)

        # Verify it was fast (should be much less than 1ms)
        duration = end_time - start_time
        self.assertLess(duration, 0.001, f"Synchronization took too long: {duration:.4f}s")

        print(f"⏱️ Synchronization completed in {duration*1000:.2f}ms")

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self.gui_app, 'camera_thread'):
            self.gui_app.camera_thread = None


def main():
    """Run the difficulty synchronization tests"""
    print("🎯 Testing Difficulty Synchronization Implementation")
    print("=" * 60)

    # Run the tests
    unittest.main(verbosity=2, exit=False)

    print("\n" + "=" * 60)
    print("✅ All difficulty synchronization tests completed!")


if __name__ == "__main__":
    main()
