def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    result = {}
    for cat, num in zip(categories, targets):
        if cat not in result:
            result[cat] = [0, 0]
        result[cat][0] += num
        result[cat][1] += 1
    return [
        result[cat][0] / result[cat][1]
        for cat in categories
    ]