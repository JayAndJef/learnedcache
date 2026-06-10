import sys
from pathlib import Path

# Put the learnedcache project root (parent of the evict_classifier package) on
# the path so `import evict_classifier` works under `python -m pytest`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
