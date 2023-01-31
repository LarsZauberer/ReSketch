import logging
from pathlib import Path
import os
from rich.logging import RichHandler
from functools import wraps
import sys
import datetime as dt


def initialize_logging(args):
    """
    initialize_logging Function to initialize the logging system

    :param args: argparse arguments
    :type args: Namespace
    """
    # Check if logs folder exists
    if not Path("logs").exists():
        os.mkdir("logs")
    
    # File Log handler
    FileHandler = logging.FileHandler(Path("logs") / Path(dt.datetime.now().strftime("%d.%m.%y %H.%M.%S.log")))
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(name)s] [%(levelname)-5.5s] %(message)s")
    FileHandler.setFormatter(logFormatter)
    
    if args.debug:
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(), FileHandler]
        )
    else:
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(), FileHandler]
        )


def critical(originalFunction):
    """Creates a wrapper function that executes every function with a try
    statement.
    Args: originalFunction (function): The function it should execute in the try statement.
    """
    @wraps(originalFunction)
    def wrapperFunction(*args, **kwargs):
        log = logging.getLogger(str(originalFunction.__qualname__))
        try:
            result = originalFunction(*args, **kwargs)
            return result
        except Exception:
            log.exception(f"Error while executing function {str(originalFunction.__qualname__)}")
            sys.exit("Error while Executing -> Exit with failure")
    return wrapperFunction
