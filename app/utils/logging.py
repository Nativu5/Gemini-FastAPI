import inspect
import logging
import sys
from typing import Literal

from loguru import logger


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
    diagnose: bool = True,
    backtrace: bool = True,
    colorize: bool = True,
) -> None:
    """
    Setup loguru logging configuration to unify all project logging output

    Args:
        level: Log level
        diagnose: Whether to enable diagnostic information
        backtrace: Whether to enable backtrace information
        colorize: Whether to enable colors
    """
    # Reset all logger handlers
    logger.remove()

    # Add unified handler for all logs
    logger.add(
        sys.stderr,
        level=level,
        colorize=colorize,
        backtrace=backtrace,
        diagnose=diagnose,
        enqueue=True,
    )

    # Setup standard logging library interceptor
    _setup_logging_intercept()

    logger.debug("Logger initialized.")


def _setup_logging_intercept() -> None:
    """Setup standard logging library interceptor to redirect to loguru"""

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if it exists.
            try:
                level: str | int = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message.
            frame, depth = inspect.currentframe(), 0
            while frame:
                filename = frame.f_code.co_filename
                is_logging = filename == logging.__file__
                is_frozen = "importlib" in filename and "_bootstrap" in filename
                if depth > 0 and not (is_logging or is_frozen):
                    break
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Remove all existing handlers and add our interceptor
    logging.basicConfig(handlers=[InterceptHandler()], level="INFO", force=True)
