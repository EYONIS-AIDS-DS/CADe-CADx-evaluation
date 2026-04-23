"""Logging configuration for the CADe/CADx evaluation package.

A single module-level :data:`logger` instance is created at import time and
writes to both *stdout* and to ``data/log/CADe_CADx_evaluate.log`` at the
repository root.  The directory is created automatically if it does not exist.
"""
import logging
from pathlib import Path


def setup_logger(logfile, level=logging.INFO):
    """Create and configure a logger that writes to file and stdout.

    Parameters
    ----------
    logfile:
        Path to the log file.
    level:
        Logging level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger("CADe_CADx_Evaluate")
    logger.setLevel(level)

    # Create a file handler and set the level
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level)

    # Create a stream handler to print to stdout and set the level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    # Create a formatter and set the format for log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the file handler and stream handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# Anchor to this file's location so the path is correct regardless of CWD.
# logger.py lives at  <repo_root>/CADe_CADx_evaluation/evaluate_common/logger.py
# so parents[2] is the repository root.
_repo_root = Path(__file__).resolve().parents[2]
_log_dir = _repo_root / "data" / "log"
_log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(_log_dir / "CADe_CADx_evaluate.log")
