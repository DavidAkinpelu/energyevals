import json
import os
from typing import Any, Optional

import pandas as pd
import requests
from loguru import logger

from energbench.agent.providers import ToolDefinition
from energbench.utils import generate_timestamp

from .base_tool import BaseTool
from .constants import GRID_STATUS_PAGE_SIZE, HTTP_TIMEOUT_EXTENDED


class GridStatusAPITool(BaseTool):
    """Tool for getting data from the GridStatus API.

    Provides access to real-time and historical data from various
    ISOs/RTOs including ERCOT, PJM, NYISO, CAISO, and more.
    """

    BASE_URL = "https://api.gridstatus.io/v1"

    def __init__(self, api_key: Optional[str] = None):
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

        self.register_method("list_gridstatus_datasets", self.list_gridstatus_datasets)
        self.register_method("inspect_gridstatus_dataset", self.inspect_gridstatus_dataset)
        self.register_method("query_gridstatus_dataset", self.query_gridstatus_dataset)

    def _make_request(
        self, endpoint: str, params: Optional[dict[str, Any]] = None
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

    def list_gridstatus_datasets(self) -> str:
        """Returns JSON list of available datasets with id, name, and description."""
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

    def inspect_gridstatus_dataset(self, dataset_id: str) -> str:
        """
        Args:
            dataset_id: The id of the dataset to inspect

        Returns:
            A JSON string of the full metadata associated with the dataset
        Prints the full metadata associated with a grid status dataset id as a JSON string.
        """
        result = self._make_request(f"datasets/{dataset_id}")
        return json.dumps(result, indent=2, default=str)

    def query_gridstatus_dataset(
        self,
        dataset_id: str,
        filter_column: Optional[str] = None,
        filter_value: Optional[str] = None,
        limit: Optional[int] = None,
        columns: Optional[list[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        time: Optional[str] = None,
        publish_time: Optional[str] = None,
        publish_time_start: Optional[str] = None,
        publish_time_end: Optional[str] = None,
        resample_frequency: Optional[str] = None,
        resample_by: Optional[str] = None,
        resample_function: Optional[str] = "mean",
    ) -> str:
        """
        Query data from a specific Grid Status dataset.
        On success (status 200), saves 'data' as CSV, prints df.head() and filepath.
        On error, prints full JSON response.
        filter operator is fixed to '=' and cannot be changed
        time comparison is fixed to '=' and cannot be changed

        Args:
        dataset_id (str): The ID of the dataset to query.
        filter_column (str): Column name to filter results by. Default is None
        filter_value (str): Value to filter results by. Default is None
        limit (int): Maximum number of rows to return. Default is None
        columns (list[str]): Columns to return. Default is None
        start_time (str): ISO 8601 start time (for time_index_column). Default is None
        end_time (str): ISO 8601 end time (for time_index_column). Default is None
        time (str): 'latest' or ISO 8601 timestamp. Default is None
        publish_time (str): Advanced filtering for forecast datasets. Default is None
        publish_time_start (str): Start of publish_time filter. Default is None
        publish_time_end (str): End of publish_time filter. Default is None
        resample_frequency (str): e.g. '1 minute', '5 minutes', '1 hour', etc. Default is None
        resample_by (str): Columns to group by before resampling. Default is None
        resample_function (str): Should be one of the following options - 'mean', 'sum', 'min', 'max', 'count', 'stddev', 'variance'. Default is 'mean'

        time cannot be used with start_time or end_time
        time_comparison can only be used when time is used
        for time related filters, NEVER use filter_column & filter_value. Use either time or start_time and end_time

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
        if time is not None:
            params["time"] = time
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

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for Grid Status tools."""
        return [
            ToolDefinition(
                name="list_gridstatus_datasets",
                description=(
                    "List all available datasets from the Grid Status API. "
                    "Returns dataset IDs, names, and descriptions for electricity "
                    "market data including prices, load, generation, and forecasts."
                ),
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="inspect_gridstatus_dataset",
                description=(
                    "Get detailed metadata for a specific dataset including "
                    "available columns, time ranges, and data frequency."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "The dataset ID (e.g., 'ercot_spp_real_time')",
                        },
                    },
                    "required": ["dataset_id"],
                },
            ),
            ToolDefinition(
                name="query_gridstatus_dataset",
                description=(
                    "Query data from a Grid Status dataset. Supports time filtering, "
                    "column selection, and resampling. On success, saves data as CSV "
                    "and prints the filepath. On error, prints the full JSON response."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "The dataset ID to query",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "ISO 8601 start time (e.g., '2024-01-01T00:00:00Z')",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "ISO 8601 end time",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of columns to return",
                        },
                        "filter_column": {
                            "type": "string",
                            "description": "Column to filter by",
                        },
                        "filter_value": {
                            "type": "string",
                            "description": "Value to filter by",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum rows to return",
                        },
                        "resample_frequency": {
                            "type": "string",
                            "description": "Resample frequency (e.g., '1 hour', '1 day')",
                        },
                        "resample_by": {
                            "type": "string",
                            "description": "Columns to group by before resampling",
                        },
                        "resample_function": {
                            "type": "string",
                            "description": "Resample function (mean, sum, min, max, count, stddev, variance)",
                            "default": "mean",
                        },
                        "time": {
                            "type": "string",
                            "description": "Specific time or 'latest' (use instead of start/end)",
                        },
                        "publish_time": {
                            "type": "string",
                            "description": "Publish time filter for forecast datasets",
                        },
                        "publish_time_start": {
                            "type": "string",
                            "description": "Publish time start filter",
                        },
                        "publish_time_end": {
                            "type": "string",
                            "description": "Publish time end filter",
                        },
                    },
                    "required": ["dataset_id"],
                },
            ),
        ]
