def clamp(value, lower, upper):
    """Return value bounded by lower and upper."""
    return min(lower, max(upper, value))
