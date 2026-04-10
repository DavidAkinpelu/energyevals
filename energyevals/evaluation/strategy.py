VALID_STRATEGIES = {"accuracy", "attributes"}


def get_strategy(
    category: str,
    category_strategies: dict[str, str],
    default_strategy: str = "attributes",
) -> str:
    """Return the judge strategy for a question category.

    Looks up *category* in the caller-provided *category_strategies* map.
    Returns *default_strategy* when the category has no explicit mapping.
    """
    return category_strategies.get(category, default_strategy)


def has_strategy(category: str, category_strategies: dict[str, str]) -> bool:
    """Return whether *category* has an explicit strategy mapping."""
    return category in category_strategies
