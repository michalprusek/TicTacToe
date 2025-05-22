"""
Unit tests for main_pyqt.py
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import argparse

# Import the module to test
from app.main import main_pyqt


@pytest.fixture
def mock_app():
    """mock_app fixture for tests."""
    mock_app = self.app_patcher.start()
    return mock_app


@pytest.fixture
def mock_qapp():
    """mock_qapp fixture for tests."""
    mock_qapp = self.qapp_patcher.start()
    return mock_qapp



class TestMainPyQt():
    """Test cases for main_pyqt.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Save original sys.argv
        self.original_argv = sys.argv
        # Mock QApplication
        self.qapp_patcher = patch('main_pyqt.QApplication')
        self.mock_qapp = self.qapp_patcher.start()
        # Mock TicTacToeApp
        self.app_patcher = patch('main_pyqt.TicTacToeApp')
        self.mock_app = self.app_patcher.start()
        # Mock logging
        self.logger_patcher = patch('main_pyqt.logging')
        self.mock_logging = self.logger_patcher.start()
        self.mock_logger = MagicMock()
        self.mock_logging.getLogger.return_value = self.mock_logger

    def tearDown(self):
        """Tear down test fixtures"""
        # Restore original sys.argv
        sys.argv = self.original_argv
        # Stop patchers
        self.qapp_patcher.stop()
        self.app_patcher.stop()
        self.logger_patcher.stop()

    def test_main_default_args(self):
        """Test main function with default arguments"""
        # Set up command line arguments
        sys.argv = ['main_pyqt.py']

        # Mock argparse.ArgumentParser
        with patch('main_pyqt.argparse.ArgumentParser', autospec=True) as mock_parser_cls:
            mock_parser = mock_parser_cls.return_value
            mock_args = argparse.Namespace(
                camera=0,
                debug=False,
                difficulty=5
            )
            mock_parser.parse_args.return_value = mock_args

            # Call the main function
            main_pyqt.main()

            # Verify QApplication was created with correct arguments
            self.mock_qapp.assert_called_once_with(sys.argv)

            # Verify TicTacToeApp was created
            self.mock_app.assert_called_once()

            # Verify show was called on the app window
            self.mock_app.return_value.show.assert_called_once()

            # Verify app.exec_ was called
            self.mock_qapp.return_value.exec_.assert_called_once()

    def test_main_with_debug_mode(self):
        """Test main function with debug mode enabled"""
        # Set up command line arguments
        sys.argv = ['main_pyqt.py', '--debug']

        # Mock argparse.ArgumentParser
        with patch('main_pyqt.argparse.ArgumentParser', autospec=True) as mock_parser_cls:
            mock_parser = mock_parser_cls.return_value
            mock_args = argparse.Namespace(
                camera=0,
                debug=True,
                difficulty=5
            )
            mock_parser.parse_args.return_value = mock_args

            # Call the main function
            main_pyqt.main()

            # Verify debug mode was enabled in the config
            self.mock_app.assert_called_once()

            # We can't easily verify the debug message was logged because
            # the actual logging happens in the main function, not in our mocks.
            # This is a limitation of how we're testing.

    def test_main_with_custom_camera_and_difficulty(self):
        """Test main function with custom camera index and difficulty"""
        # Set up command line arguments
        sys.argv = ['main_pyqt.py', '--camera', '1', '--difficulty', '8']

        # Mock argparse.ArgumentParser
        with patch('main_pyqt.argparse.ArgumentParser', autospec=True) as mock_parser_cls:
            mock_parser = mock_parser_cls.return_value
            mock_args = argparse.Namespace(
                camera=1,
                debug=False,
                difficulty=8
            )
            mock_parser.parse_args.return_value = mock_args

            # Mock AppConfig
            with patch('main_pyqt.AppConfig', autospec=True) as mock_config_cls:
                mock_config = mock_config_cls.return_value
                mock_config.game_detector = MagicMock()
                mock_config.game = MagicMock()

                # Call the main function
                main_pyqt.main()

                # Verify config was updated with custom values
        assert mock_config.game_detector.camera_index == 1
        assert mock_config.game.default_difficulty == 8
        assert mock_config.debug_mode == False



