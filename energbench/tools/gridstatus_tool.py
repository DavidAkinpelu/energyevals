import json
import os
from typing import Any, Literal

import pandas as pd
import requests
from loguru import logger

from energbench.utils import generate_timestamp

from .base_tool import BaseTool, tool_method
from .constants import GRID_STATUS_PAGE_SIZE, HTTP_TIMEOUT_EXTENDED


class GridStatusAPITool(BaseTool):
    """Tool for getting data from the GridStatus API.

    Provides access to real-time and historical data from various
    ISOs/RTOs including ERCOT, PJM, NYISO, CAISO, and more.
    """

    BASE_URL = "https://api.gridstatus.io/v1"

    def __init__(self, api_key: str | None = None):
        """Initialize the Grid Status tool.

        Args:
            api_key: Grid Status API key. Defaults to GRIDSTATUS_API_KEY env var.
        """
        super().__init__(
            name="gridstatus_api_tool",
            description="Access electricity market data from Grid Status API",
        )

        self.api_key = api_key or os.getenv("GRIDSTATUS_API_KEY")

        if not self.api_key:
            logger.warning("GRIDSTATUS_API_KEY not set. Tool will not function.")

    def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a request to the Grid Status API."""
        if not self.api_key:
            return {"error": "GRIDSTATUS_API_KEY not configured"}

        headers = {"x-api-key": self.api_key}
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT_EXTENDED)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Grid Status API request failed: {e}")
            return {"error": str(e)}

    def set_api_key(self, api_key: str) -> None:
        """Set the API key for Grid Status API."""
        self.api_key = api_key

    @tool_method()
    def list_gridstatus_datasets(self) -> str:
        """Return a JSON list of available Grid Status datasets with each dataset's id, name, and description.
        Use this first to discover valid dataset IDs before inspection or querying."""

        result = self._make_request("datasets")

        if "error" in result:
            return json.dumps(result)

        datasets = [
            {
                "id": ds["id"],
                "name": ds["name"],
                "description": ds.get("description", ""),
            }
            for ds in result.get("data", [])
        ]

        return json.dumps(datasets, indent=2)

    @tool_method()
    def inspect_gridstatus_dataset(self, dataset_id: str) -> str:
        """Return full metadata for a specific Grid Status dataset to understand its schema and query options before running dataset queries.

        Args:
            dataset_id: The id of the gridstatus dataset to inspect.

        Returns:
            A JSON string of the full metadata associated with the dataset.
        """
        result = self._make_request(f"datasets/{dataset_id}")
        return json.dumps(result, indent=2, default=str)

    @tool_method()
    def query_gridstatus_dataset(
        self,
        dataset_id: str,
        filter_column: str | None = None,
        filter_value: str | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        time_val: str | None = None,
        publish_time: str | None = None,
        publish_time_start: str | None = None,
        publish_time_end: str | None = None,
        resample_frequency: str | None = None,
        resample_by: str | None = None,
        resample_function: Literal["mean", "sum", "min", "max", "count", "stddev", "variance"] | None = "mean",
    ) -> str:
        """Query data from a Grid Status dataset with optional filtering, column selection, and resampling.
        On success, saves data to CSV and returns preview rows, filepath, and row count; on error, returns
        the full JSON error. Notes: filter_operator and time_comparison are fixed to '=' and cannot be
        changed. time_val cannot be used together with start_time or end_time. For time-related filters,
        NEVER use filter_column and filter_value; use start_time/end_time or time_val instead. If an input
        is not used, set it to None or an empty string. Also important to note that each query can only return
        up to 50,000 rows at a time so it is important to split the queries when data pulls over longer periods
        are required. The resample inputs are also very powerful and you should use them often when calculating
        data aggregations

        Args:
            dataset_id: The ID of the dataset to query.
            filter_column: Column name to filter results by. Do NOT use for time-related filtering.
            filter_value: Value to filter results by. Do NOT use for time-related filtering.
            limit: Maximum number of rows to return.
            columns: Columns to return.
            start_time: ISO 8601 start time (for time_index_column). Cannot be used with time_val.
            end_time: ISO 8601 end time (for time_index_column). Cannot be used with time_val.
            time_val: 'latest' or ISO 8601 timestamp. Cannot be used with start_time or end_time.
            publish_time: Advanced filtering for forecast datasets.
            publish_time_start: Start of publish_time filter.
            publish_time_end: End of publish_time filter.
            resample_frequency: e.g. '1 minute', '5 minutes', '1 hour', etc.
            resample_by: Columns to group by before resampling.
            resample_function: One of 'mean', 'sum', 'min', 'max', 'count', 'stddev', 'variance'. Default is 'mean'.

        Returns:
            JSON string. On success contains 'preview' (first rows as list of dicts), 'filepath'
            (absolute path to saved CSV), and 'row_count'. On error contains the full API error response.
        """
        page = 1
        page_size = GRID_STATUS_PAGE_SIZE
        order = "asc"
        return_format = "json"
        timezone = "market"
        filter_operator = "="
        time_comparison = "="

        params = {
            "order": order,
            "return_format": return_format,
            "filter_operator": filter_operator,
            "time_comparison": time_comparison,
            "timezone": timezone,
            "page": page,
            "page_size": page_size,
        }

        if filter_column is not None:
            params["filter_column"] = filter_column
        if filter_value is not None:
            params["filter_value"] = filter_value
        if limit is not None:
            params["limit"] = limit
        if columns is not None:
            params["columns"] = ",".join(columns)
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        if time_val is not None:
            params["time"] = time_val
        if publish_time is not None:
            params["publish_time"] = publish_time
        if publish_time_start is not None:
            params["publish_time_start"] = publish_time_start
        if publish_time_end is not None:
            params["publish_time_end"] = publish_time_end
        if resample_frequency is not None:
            params["resample_frequency"] = resample_frequency
        if resample_by is not None:
            params["resample_by"] = resample_by
        if resample_function is not None:
            params["resample_function"] = resample_function

        params = {k: v for k, v in params.items() if v not in ("", None)}

        result = self._make_request(f"datasets/{dataset_id}/query", params)

        if "error" in result:
            return json.dumps(result, indent=2)

        data = result.get("data", [])

        if data:
            try:
                df = pd.DataFrame(data)
                timestamp = generate_timestamp()
                filename = f"{dataset_id}_{timestamp}.csv"
                filepath = os.path.abspath(filename)
                df.to_csv(filepath, index=False)

                output = {
                    "preview": df.head().to_dict("records"),
                    "filepath": filepath,
                    "row_count": len(df),
                }
                return json.dumps(output, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save CSV: {e}")
                return json.dumps({"error": str(e), "data": data}, indent=2, default=str)
        else:
            return json.dumps({"message": "No data found in response", "response": result}, indent=2, default=str)
