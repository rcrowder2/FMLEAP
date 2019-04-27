

def area_under_curve(x, y):
    """
    Computes the AUC of a set of points, using an approach analogous to a left Riemann sum.

    >>> steps = [0, 1, 2, 3, 4]
    >>> fitnesses = [0, 0.5, 0.9, 0.99, 1.0]
    >>> round(area_under_curve(steps, fitnesses), 2)
    2.39

    Step sizes can vary:

    >>> x = [0, 10, 20, 30, 50, 100]
    >>> y = [0.2, 0.5, 0.9, 1.2, 1.3, 1.4]
    >>> round(area_under_curve(x, y), 2)
    105.0
    """
    assert(len(x) == len(y)), f"x and y must be of the same length (found {len(x)} and {len(y)}, respectively"
    auc = 0
    for i in range(len(x) - 1):
        step_size = (x[i+1] - x[i])
        auc += y[i] * step_size
    return auc
