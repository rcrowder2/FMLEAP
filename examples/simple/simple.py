"""
    This is the simplest example of using LEAP, where one can rely on the
    very high-level function, ea_solve(), to optimize the given real-valued
    function.
"""
from leap_ec.simple import ea_solve


def function(values):
    return sum([x ** 2 for x in values])

if __name__ == '__main__':
    ea_solve(function,
            bounds=[(-5.12, 5.12) for _ in range(5)],
            viz=True,
            mutation_std=0.1)