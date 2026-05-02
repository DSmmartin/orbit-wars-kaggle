"""Entry point for PPO training — suppresses noisy third-party logs before any orbit_wars import."""

import io
import logging
import sys

# logging.disable blocks at the global manager level before any child logger
# can override its own level (open_spiel_env forcibly sets itself to INFO).
logging.disable(logging.WARNING)

# kaggle_environments/__init__.py prints cabt load failures to stdout at import time.
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from orbit_wars.academy.campaign import main
finally:
    sys.stdout = _real_stdout

# Silence loguru's default stderr handler so observatory timed_calc
# and sniper debug calls don't leak through during training.
from loguru import logger as _logger  # noqa: E402
_logger.remove()
_logger.add(sys.stderr, level="ERROR")

if __name__ == "__main__":
    main()
