def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    return len(set([
        item
        for items in recommendations
        for item in items
    ])) / n_items