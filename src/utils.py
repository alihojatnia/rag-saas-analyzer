import logging
from datetime import datetime

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("rag-demo")


# -------------------------------------------------
# Simple tool: current date
# -------------------------------------------------
def get_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")