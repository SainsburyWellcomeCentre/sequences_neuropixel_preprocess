import deeplabcut as dlc

import sys
from pathlib import Path

from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback

install_traceback()

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)

logger.info('arguments:')
logger.info(sys.argv)

for arg in sys.argv[1:]:
    if not Path(arg).exists():
        raise FileNotFoundError(arg)

logger.info('All checks passed, starting tracking for real now')
dlc.analyze_videos(sys.argv[1], [str(sys.argv[2])], destfolder=sys.argv[3])
logger.info('Done, hopefully...')
