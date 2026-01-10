#!/usr/bin/env python
"""Exercise GridStatusAPITool methods."""
from __future__ import annotations

import json

from dotenv import load_dotenv

from energbench.tools.gridstatus_tool import GridStatusAPITool


def _print_result(label: str, result: str) -> None:
    print(f"\n=== {label} ===")
    print(result)


def _first_dataset_id(result: str) -> str | None:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first.get("id")
    return None


def main() -> None:
    load_dotenv()
    tool = GridStatusAPITool()

    list_result = tool.list_gridstatus_datasets()
    _print_result("list_gridstatus_datasets", list_result)

    dataset_id = _first_dataset_id(list_result)
    if not dataset_id:
        print("\nNo dataset id available for inspect/query.")
        return

    # _print_result(
    #     "inspect_gridstatus_dataset",
    #     tool.inspect_gridstatus_dataset(dataset_id=dataset_id),
    # )
    _print_result(
        "query_gridstatus_dataset",
        tool.query_gridstatus_dataset(dataset_id=dataset_id, limit=1),
    )


if __name__ == "__main__":
    main()
