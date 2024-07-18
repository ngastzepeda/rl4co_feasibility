import logging
import sys
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger as loguru_logger


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Set levels for handlers
    stdout_handler.setLevel(logging.DEBUG)  # Handle DEBUG and above levels to INFO
    stderr_handler.setLevel(logging.WARNING)  # Handle WARNING and above levels

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s\t | %(name)s - %(message)s"
    )
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    # This ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# Initialize the standard logger
std_logger = get_pylogger()


# Intercept logs from the standard logging module to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get the corresponding Loguru level if it exists
        level = (
            loguru_logger.level(record.levelname).name
            if record.levelname in loguru_logger._core.levels
            else record.levelno
        )
        # Find the caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Clear the existing root handlers
logging.root.handlers = []
logging.basicConfig(handlers=[InterceptHandler()], level=0)

# Now use loguru for logging
loguru_logger.remove()  # Remove any default handler if it exists
loguru_logger.add(sys.stdout, level="INFO")  # Log INFO and above to stdout
loguru_logger.add(sys.stderr, level="ERROR")  # Log ERROR and above to stderr
