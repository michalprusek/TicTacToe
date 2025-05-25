"""
Path utilities for the TicTacToe application.
Consolidates repeated path setup patterns to avoid duplication.
"""

import sys
import os


def setup_project_path():
    """
    Setup project root in sys.path for imports.
    Used across multiple modules to ensure consistent path handling.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def setup_uarm_sdk_path():
    """
    Setup uArm SDK path for arm controller.
    Returns the uArm SDK path if it exists.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    uarm_sdk_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "uArm-Python-SDK")

    if os.path.exists(uarm_sdk_path):
        if uarm_sdk_path not in sys.path:
            sys.path.insert(0, uarm_sdk_path)
        return uarm_sdk_path
    return None


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def get_weights_path():
    """Get the weights directory path."""
    return os.path.join(get_project_root(), "weights")


def get_calibration_path():
    """Get the calibration file path."""
    return os.path.join(get_project_root(), "app", "calibration")