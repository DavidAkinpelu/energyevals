#!/usr/bin/env python
"""Exercise OpenWeatherTool methods."""
from __future__ import annotations

import time

from dotenv import load_dotenv

from energbench.tools.openweather_tool import OpenWeatherTool


def _print_result(label: str, result: str) -> None:
    print(f"\n=== {label} ===")
    print(result)


def main() -> None:
    load_dotenv()
    tool = OpenWeatherTool()

    location = "Austin, TX, USA"
    _print_result("geocode_location", tool.geocode_location(location=location, limit=1))
    _print_result("get_current_weather", tool.get_current_weather(location=location))
    _print_result("get_forecast", tool.get_forecast(location=location, days=2))

    now = int(time.time())
    start = now - 6 * 3600
    end = now - 3600
    _print_result(
        "get_historical_weather",
        tool.get_historical_weather(location=location, start=start, end=end, type_inp="hour"),
    )
    _print_result("get_air_pollution", tool.get_air_pollution(location=location))


if __name__ == "__main__":
    main()
