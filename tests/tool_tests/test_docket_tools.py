#!/usr/bin/env python
"""Exercise DocketTools methods."""
from __future__ import annotations

from dotenv import load_dotenv

from energbench.tools.docket_tools import DocketTools


def _print_result(label: str, result: str) -> None:
    print(f"\n=== {label} ===")
    print(result)


def main() -> None:
    load_dotenv()
    tool = DocketTools()

    # _print_result(
    #     "search_ferc_dockets",
    #     tool.search_ferc(
    #         start_date="2024-01-01",
    #         end_date="2024-01-07",
    #         keyword="solar",
    #         results_per_page=5,
    #         page=0,
    #     ),
    # )
    # _print_result(
    #     "get_maryland_psc_item",
    #     tool.get_maryland_psc_item(kind="case", number="0", timeout=15),
    # )
    # _print_result(
    #     "get_maryland_official_filings",
    #     tool.get_maryland_official_filings(
    #         start_date="01/01/2024",
    #         end_date="01/07/2024",
    #         timeout=15,
    #     ),
    # )
    # _print_result(
    #     "search_texas_dockets",
    #     tool.search_texas(
    #         date_from="01/01/2024",
    #         date_to="01/07/2024",
    #         timeout=15,
    #     ),
    # )
    # _print_result(
    #     "search_new_york_dockets",
    #     tool.search_new_york(
    #         start_date="01/01/2024",
    #         end_date="01/07/2024",
    #         mode="cases",
    #         timeout=15,
    #     ),
    # )
    # _print_result(
    #     "search_north_carolina_dockets",
    #     tool.search_north_carolina(
    #         date_from="01/01/2024",
    #         date_to="01/07/2024",
    #         max_pages=1,
    #         timeout=15,
    #     ),
    # )
    # _print_result(
    #     "search_south_carolina_dockets",
    #     tool.search_south_carolina(
    #         start_date="2024-01-01",
    #         end_date="2024-01-07",
    #         timeout=15,
    #     ),
    # )
    # _print_result(
    #     "search_virginia_dockets",
    #     tool.search_virginia(
    #         start_date="2024-01-01",
    #         end_date="2024-01-07",
    #         timeout=15,
    #     ),
    # )
    # _print_result(
    #     "search_dc_dockets",
    #     tool.search_dc(
    #         start_date="01/01/2024",
    #         end_date="01/07/2024",
    #         records_to_show=5,
    #         timeout=15,
    #     ),
    # )


if __name__ == "__main__":
    main()
