MAX_ITERATIONS: int = 25 # Maximum number of ReAct iterations before the agent stops.

CSV_THRESHOLD: int = 20 # Row count threshold for saving tool results to CSV instead of inline.

MAX_TOKENS: int = 4096 # Default maximum tokens for LLM completion responses.

PREVIEW_ROWS: int = 5 # Number of preview rows to include when large results are saved to CSV.

QUERY_TRUNCATE_LENGTH: int = 100 # Maximum characters shown when logging a query.

TOOL_TIMEOUT: float = 60.0 # Seconds before a stalled tool call is cancelled.

PROVIDER_MAX_RETRIES: int = 3 # Maximum number of retries for provider complete() calls.

PROVIDER_RETRY_BASE_DELAY: float = 1.0 # Base delay in seconds for exponential backoff.
