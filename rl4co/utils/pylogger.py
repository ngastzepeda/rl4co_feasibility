import logging
import sys
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger as loguru_logger


def configure_wandb_logger():
    # use loguru for logging
    loguru_logger.remove()  # Remove any default handler if it exists
    loguru_logger.add(
        sys.stdout, level="INFO", filter=lambda record: record["level"].no < 40
    )
    loguru_logger.add(sys.stderr, level="ERROR")  # Log ERROR and above to stderr


loguru_logger.remove(handler_id=1)  # Remove the default stderr handler
if not loguru_logger._core.handlers:
    configure_wandb_logger()


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


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)
    logger.addHandler(InterceptHandler())
    logger.propagate = False

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
