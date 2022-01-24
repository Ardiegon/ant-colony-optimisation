import pandas as pd
import numpy as np
import logging
import time
from project.src.thread_with_return import ThreadWithReturn
from project.src.grouped_data import GroupedData

class Model:
    def __init__(self, _n_ants, _pheromone_weight,  _backpack_size, _distance_weight):
        self.n_ants = _n_ants
        self.pheromone_weight = _pheromone_weight
        self.backpack_size = _backpack_size
        self.distance_weight = _distance_weight

    def load_data(self, data: GroupedData):
        pass

if __name__ == "__main__":
    data = GroupedData()
