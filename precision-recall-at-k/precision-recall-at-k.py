def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    intersect = len(set(recommended[:k]) & set(relevant))
    return [
        intersect / k,
        intersect / len(relevant)
    ]