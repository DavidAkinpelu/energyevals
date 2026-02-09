from typing import Any, Optional

from loguru import logger

from energbench.tools.base_tool import BaseTool


class DocketBaseTool(BaseTool):
    """Base class for jurisdiction-specific docket tools.

    Provides shared helpers (e.g. CSV export) used by every docket scraper.
    """

    @staticmethod
    def _save_csv(rows: list[dict[str, Any]], save_csv_path: Optional[str]) -> Optional[str]:
        if not save_csv_path:
            return None
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_csv(save_csv_path, index=False)
            return save_csv_path
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return None
