def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    return [
        max((
            rewards[s][a] + gamma * sum((
                prob * values[s_next]
                for s_next, prob in enumerate(action)
            ))
            for a, action in enumerate(state)
        ))
        for s, state in enumerate(transitions)
    ]