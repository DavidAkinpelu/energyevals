_STRATEGY_MAP: dict[str, str] = {
    "Market data retrieval and analysis": "accuracy",
}

_DEFAULT_STRATEGY = "attributes"


def get_strategy(category: str) -> str:
    """Return the accuracy judge type for a question category.

    Returns ``"accuracy"`` for *Market data retrieval and analysis*,
    ``"attributes"`` for everything else.
    """
    return _STRATEGY_MAP.get(category, _DEFAULT_STRATEGY)


def has_strategy(category: str) -> bool:
    """Return whether *category* has an explicit strategy mapping."""
    return category in _STRATEGY_MAP
