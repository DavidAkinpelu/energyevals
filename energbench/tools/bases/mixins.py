from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from energbench.utils import HTTPClient, save_to_csv


class HTTPMixin:
    """Mixin for tools that need HTTP request capabilities.

    Provides a configured HTTP client for making API requests.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._http_client: HTTPClient | None = None

    def get_http_client(
        self,
        auth_method: str = "header",
        auth_param_name: str = "x-api-key",
        timeout: int = 30,
        retries: int = 3,
    ) -> HTTPClient:
        """Get or create HTTP client.

        Args:
            auth_method: Authentication method ("header" or "param")
            auth_param_name: Name of the header/param for the API key
            timeout: Request timeout in seconds
            retries: Number of retry attempts

        Returns:
            Configured HTTP client
        """
        if self._http_client is None:
            self._http_client = HTTPClient(
                auth_method=auth_method,
                auth_param_name=auth_param_name,
                timeout=timeout,
                retries=retries,
            )
        return self._http_client


class CSVMixin:
    """Mixin for tools that save results to CSV files.

    Provides functionality for saving pandas DataFrames to timestamped CSV files.
    """

    def save_result_to_csv(
        self,
        df: pd.DataFrame,
        prefix: str,
        output_dir: Path | str = ".",
    ) -> Path:
        """Save DataFrame to CSV with timestamped filename.

        Args:
            df: DataFrame to save
            prefix: Prefix for the filename (e.g., tool name)
            output_dir: Directory to save the file

        Returns:
            Path to the saved CSV file
        """
        filepath = save_to_csv(df, prefix, output_dir)
        logger.info(f"{self.__class__.__name__}: Saved {len(df)} rows to {filepath}")
        return filepath
