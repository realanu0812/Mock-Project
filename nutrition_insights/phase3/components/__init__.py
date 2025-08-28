# nutrition_insights/phase3/components/__init__.py
# Package aggregator for component modules.
# IMPORTANT: use relative imports so this works when cwd = phase3/
from . import header
from . import overview
from . import trending
from . import insights
from . import business
## REMOVED: from . import volume
from . import chatbot
from . import export

__all__ = [
    "header",
    "overview",
    "trending",
    "insights",
    "business",
    ## REMOVED: "volume",
    "chatbot",
    "export",
]