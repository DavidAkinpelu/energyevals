from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from loguru import logger

from energbench.agent.constants import CSV_THRESHOLD, PREVIEW_ROWS
from energbench.utils import generate_timestamp


class CSVProcessor:
    """Processor for handling large CSV results from tools.

    When tool results contain large tabular data (exceeding the row threshold),
    this processor saves the data to CSV and returns a reference instead of
    the full data.
    """

    def __init__(
        self,
        threshold: int = CSV_THRESHOLD,
        output_dir: str | Path = "./agent_outputs",
    ):
        """Initialize CSV processor.

        Args:
            threshold: Row count threshold for saving to CSV
            output_dir: Directory to save CSV files
        """
        self.threshold = threshold
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def process(
        self,
        tool_name: str,
        result: str,
    ) -> tuple[str, str | None]:
        """Process tool result, saving to CSV if it exceeds threshold.

        Args:
            tool_name: Name of the tool that produced the result
            result: Tool result as JSON string

        Returns:
            Tuple of (context_result, csv_path):
                - context_result: Result to pass to LLM (may be reference to CSV)
                - csv_path: Path to saved CSV file (None if not saved)
        """
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return result, None

        rows = None
        columns = None

        if isinstance(data, dict):
            rows = data.get("rows")
            columns = data.get("columns")

        if not rows or not isinstance(rows, list):
            return result, None

        row_count = len(rows)

        if row_count <= self.threshold:
            return result, None

        timestamp = generate_timestamp()
        csv_filename = f"{tool_name}_{timestamp}.csv"
        csv_path = self.output_dir / csv_filename

        try:
            with open(csv_path, "w", newline="") as f:
                if rows and isinstance(rows[0], dict):
                    fieldnames = list(rows[0].keys())
                    dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
                    dict_writer.writeheader()
                    dict_writer.writerows(rows)
                elif columns:
                    list_writer = csv.writer(f)
                    list_writer.writerow(columns)
                    list_writer.writerows(rows)
                else:
                    return result, None

            preview_rows = rows[:PREVIEW_ROWS]
            context_data = {
                "status": "success",
                "row_count": row_count,
                "csv_file": str(csv_path),
                "message": f"Query returned {row_count} rows. Results saved to {csv_path}. Use Python to read and analyze the CSV file.",
                "columns": columns or (list(rows[0].keys()) if rows else []),
                "preview": preview_rows,
            }

            for key in ["database", "query", "table"]:
                if key in data:
                    context_data[key] = data[key]

            return json.dumps(context_data, indent=2, default=str), str(csv_path)

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return result, None
