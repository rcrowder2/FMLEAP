
class Individual:
    def __init__(self, genome, fitness=None):
        self.genome = genome.copy()
        self.fitness = fitness

    def __repr__(self):
        return f"{self.genome}"

    def copy(self):
        return Individual(self.genome.copy())