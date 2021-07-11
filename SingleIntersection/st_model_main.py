from os import path
import mpi4py
import numpy as np
import pandas as pd
import os
from mpi4py import MPI

from PredictiveModel import PredictiveModel
from utils import  data_split

if __name__ == "__main__":

    for index in range(100):
        