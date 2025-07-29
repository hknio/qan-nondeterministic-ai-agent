import logging
import colorlog
import os
from typing import Optional


def configure_logger(
    logger_name: Optional[str] = None,
    level: Optional[int] = None,
    log_format: str = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    log_to_file: bool = False,
    log_file: str = "app.log",
) -> logging.Logger:
    """
    Configure a colorful logger with the given name and level.

    This function sets up logging with colorized console output and optional file output.

    Environment Variables:
        LOGLEVEL: Override default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_TO_FILE: If set to "1" or "true", logs will be written to file
        LOG_FILE: Path to the log file (default: "app.log")

    Args:
        logger_name: The name of the logger to configure. If None, returns the root logger.
        level: The logging level for this logger. Overridden by LOGLEVEL env var if set.
        log_format: The format string for log messages.
        date_format: The format string for log message timestamps.
        log_to_file: Whether to log to a file in addition to console.
        log_file: Path to the log file.

    Returns:
        The configured logger instance.
    """
    # Get log levels from environment if set
    if level is None:
        level = get_log_level_from_env("LOGLEVEL", logging.INFO)

    # Check if we should log to file
    log_to_file_env = os.environ.get("LOG_TO_FILE", "").lower()
    if log_to_file_env in ("1", "true", "yes"):
        log_to_file = True

    # Get log file path from environment if set
    log_file = os.environ.get("LOG_FILE", log_file)

    # Configure the logger with colorlog for console output
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            log_format,
            datefmt=date_format,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    # Get the logger
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        for h in logger.handlers:
            logger.removeHandler(h)

    # Add console handler
    logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        try:
            # Create directory for log file if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

            # Create file handler with standard formatter (no colors)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt=date_format,
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {os.path.abspath(log_file)}")
        except Exception as e:
            # Don't fail if file logging fails, just log a warning
            logger.warning(f"Failed to setup file logging: {str(e)}")

    # Set the overall logger level
    logger.setLevel(level)

    return logger


def get_log_level_from_env(env_var: str, default: int = logging.INFO) -> int:
    """
    Get log level from specified environment variable or use default.

    Args:
        env_var: Name of the environment variable to check
        default: Default log level if environment variable not set or invalid

    Returns:
        The log level as an integer
    """
    log_level_str = os.environ.get(env_var, "").upper()

    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return log_levels.get(log_level_str, default)
