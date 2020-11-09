"""Helpers for testing the output of stochastic functions."""
from scipy.stats import chisquare


def collect_distribution(function, samples: int):
    """Count the number of times the given function returns each
    output value."""
    assert(callable(function))

    outputs = {}
    for i in range(samples):
        o = function()
        outputs[o] = outputs.get(o, 0) + 1
    
    return outputs


def stochastic_chisquare(expected_distribution, distribution):
    """Use a $\chi^2$ distribution to compute a p-value for the probability of
    rejecting the hypothesis that the given distribution matches the expected
    distribution.
    
    This takes two dictionaries of values:

    >>> expected_distribution = { 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10}
    >>> distribution = { 1: 5, 2: 8, 3: 9, 4: 8, 5: 10, 6: 20}
    >>> stochastic_chisquare(expected_distribution, distribution)
    0.01990...
    """
    assert(sum(expected_distribution.values()) == sum(distribution.values())), f"The distributions have {sum(expected_distribution.values())} and {sum(distribution.values())} samples, respectively, but must be equal."
    
    def add_keys_from(dist1, dist2):
        """If dist1 contains a key that dist2 doesn't, add it to dict2."""
        for k in dist1.keys():
            if k not in dist2:
                dist2[k] = 0

    def values_sorted_by_key(dist):
        """Get the values of dist, sorted by the keys."""
        return [dist[k] for k in sorted(dist.keys())]
    
    add_keys_from(expected_distribution, distribution)
    add_keys_from(distribution, expected_distribution)

    expected_values = values_sorted_by_key(expected_distribution)
    values = values_sorted_by_key(distribution)

    _, p_value = chisquare(values, expected_values)
    return p_value


def stochastic_equals(expected_distribution, distribution, p=0.001):
    """Use a $\chi^2$ test to determine whether two discrete distributions are
    equal.

    For example, the following tests whether a 6-sided die is unbiased:
    
    >>> expected_distribution = { 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10}
    >>> distribution = { 1: 5, 2: 8, 3: 9, 4: 8, 5: 10, 6: 20}
    >>> stochastic_equals(expected_distribution, distribution)
    False
    
    """
    p_value = stochastic_chisquare(expected_distribution, distribution)
    return p_value <= p




