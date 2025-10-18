import logging
from pathlib import Path


def setup_logger(logfile, level=logging.INFO):
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


path = Path().resolve().parents[1]
logger = setup_logger(path / "CADe_CADx_evaluate.log")
