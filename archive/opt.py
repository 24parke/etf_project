import yfinance as yf
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class Summary():
    def __init__(self, u, t, start, end):
        self.u = u
        self.t = t
        self.start = start
        self.end = end
        self.u_value = 3000
        self.t_value = 1000
