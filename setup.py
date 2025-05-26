#!/usr/bin/env python3
# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""Setup script for TicTacToe robotic application."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="robotic-tictactoe",
    version="1.0.0",
    author="TicTacToe Robot Team",
    description="Robotic TicTacToe application with computer vision and AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tictactoe-robot=app.main.main_pyqt:main",
            "tictactoe-calibrate=app.calibration.calibration:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": [
            "calibration/*.json",
            "../weights/*.pt",
        ],
    },
)
