#!/usr/bin/env python
"""Exercise TariffsTool (OpenEI)."""
from dotenv import load_dotenv

from energbench.tools.tariffs_tool import TariffsTool


def _print_result(label: str, result: str) -> None:
    print(f"\n=== {label} ===")
    print(result)


def main() -> None:
    load_dotenv()
    tool = TariffsTool()
    result = tool.get_utility_tarriffs(
        address="1600 Pennsylvania Ave NW, Washington, DC 20500",
        sector="Commercial",
    )
    _print_result("get_utility_tariffs", result)


if __name__ == "__main__":
    main()
