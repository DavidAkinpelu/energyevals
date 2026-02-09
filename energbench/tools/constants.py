HTTP_TIMEOUT_EXTENDED: int = 60 # Extended timeout for slower APIs (GridStatus, FERC, etc.).

HTTP_TIMEOUT_LONG: int = 120 # Long timeout for very slow APIs (Renewables.ninja).

GRID_STATUS_PAGE_SIZE: int = 50000 # Default page size for GridStatus API pagination.

SEARCH_TEXT_LIMIT: int = 1000 # Maximum characters of text content per search result.

SEARCH_MAX_RESULTS: int = 10 # Maximum number of search results allowed.

SYSTEM_MAX_RESULTS: int = 200 # Default maximum results for file listing and grep operations.

SYSTEM_COMMAND_TIMEOUT: int = 60 # Default timeout in seconds for shell command execution.

BATTERY_ROUNDING_PROFILE: int = 4 # Decimal places for battery operation profile values.

BATTERY_ROUNDING_SUMMARY: int = 2 # Decimal places for battery summary revenue metrics.

BATTERY_INITIAL_SOC_FRACTION: float = 0.5 # Initial state of charge as fraction of capacity (0.5 = 50%).

DATA_PREVIEW_SIZE: int = 10 # Number of sample data items to include in API response previews.

CSV_PREVIEW_ROWS: int = 5 # Number of rows to preview when displaying CSV data.
