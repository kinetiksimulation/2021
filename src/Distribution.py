from scipy.stats import norm
import random


class Distribution:
    def __init__(self):
        pass

    def inverse(self) -> float:
        pass


class Normal(Distribution):
    def __init__(self, std_dev: float, mean: float):
        super(Normal, self).__init__()
        self.std_dev = std_dev
        self.mean = mean

    def inverse(self) -> float:
        return abs(norm.ppf(random.random(), loc=self.mean, scale=self.std_dev))
