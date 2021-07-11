from os import path
import numpy as np
import pandas as pd
import os

from PredictiveModel import PredictiveModel
from utils import data_split, read_data_from_xml


if __name__ == "__main__":
     
    adjacency_matrix = [[0, 750, 750, 750, 750],
                        [750, 0, 0, 0, 0],
                        [750, 0, 0, 0, 0],
                        [750, 0, 0, 0, 0],
                        [750, 0, 0, 0, 0],]

    x_train = y_train = x_val = y_val = np.zeros(100)

    for index in range(100):
        #reading data form files
        print('----- Readinding from data%i -----' % (index))
        file_name = 'data' + str(index) + '.xml'
        traffic_data = []
        traffic_data.append(read_data_from_xml(file_name,index))


