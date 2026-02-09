"""
Logging utilities for SG-MRM project.

Provides consistent logging configuration across all modules.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(name: str = 'sg_mrm',
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level (default INFO)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'sg_mrm') -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class SimulationLogger:
    """
    Specialized logger for simulation progress tracking.

    Provides methods for logging simulation milestones, layer computations,
    and performance metrics.
    """

    def __init__(self, name: str = 'sg_mrm.simulation'):
        """
        Initialize simulation logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self.start_time = None
        self.layer_times = {}

    def start_simulation(self, config: dict):
        """
        Log simulation start.

        Args:
            config: Simulation configuration dictionary
        """
        self.start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("Starting SG-MRM Simulation")
        self.logger.info(f"Origin: ({config.get('origin_lat')}, {config.get('origin_lon')})")
        self.logger.info(f"Frequency: {config.get('frequency_ghz')} GHz")
        self.logger.info(f"Start time: {self.start_time}")
        self.logger.info("=" * 60)

    def end_simulation(self):
        """Log simulation end and total duration."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            self.logger.info("=" * 60)
            self.logger.info("Simulation Complete")
            self.logger.info(f"Total duration: {duration}")
            self.logger.info("=" * 60)

    def log_layer_start(self, layer_name: str, timestamp: datetime):
        """
        Log layer computation start.

        Args:
            layer_name: Name of the layer
            timestamp: Simulation timestamp
        """
        self.layer_times[layer_name] = datetime.now()
        self.logger.info(f"Computing {layer_name} for timestamp {timestamp}")

    def log_layer_end(self, layer_name: str):
        """
        Log layer computation end.

        Args:
            layer_name: Name of the layer
        """
        if layer_name in self.layer_times:
            duration = datetime.now() - self.layer_times[layer_name]
            self.logger.info(f"{layer_name} completed in {duration.total_seconds():.2f}s")

    def log_progress(self, current: int, total: int, message: str = ""):
        """
        Log progress for iterative operations.

        Args:
            current: Current iteration
            total: Total iterations
            message: Optional message
        """
        percentage = (current / total) * 100
        self.logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) {message}")

    def log_error(self, error: Exception, context: str = ""):
        """
        Log an error with context.

        Args:
            error: Exception object
            context: Additional context information
        """
        self.logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
