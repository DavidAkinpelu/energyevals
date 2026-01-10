#!/usr/bin/env python
"""Exercise RenewablesTool methods."""
from __future__ import annotations

from dotenv import load_dotenv

from energbench.tools.renewables_tool import RenewablesTool


def _print_result(label: str, result: str) -> None:
    print(f"\n=== {label} ===")
    print(result)


def main() -> None:
    load_dotenv()
    tool = RenewablesTool()

    _print_result(
        "get_solar_profile",
        tool.get_solar_profile(
            lat=30.2672,
            lon=-97.7431,
            date_from="2019-01-01",
            date_to="2019-01-02",
            capacity=1.0,
        ),
    )
    _print_result(
        "get_wind_profile",
        tool.get_wind_profile(
            lat=30.2672,
            lon=-97.7431,
            date_from="2019-01-01",
            date_to="2019-01-02",
            capacity=1.0,
        ),
    )


if __name__ == "__main__":
    main()
