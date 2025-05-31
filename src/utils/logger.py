import logging
import sys

def setup_logger():
    """
    Sets up a logger with a standard format and INFO level.
    """
    logger = logging.getLogger("LLM-Ticket-Recommender")
    logger.setLevel(logging.INFO)
    
    # Check if a handler already exists (to avoid duplicate logs)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
