import random
import numpy as np

def set_seed_for_random(s: int = 2137):
    random.seed(s)
    np.random.seed(s)