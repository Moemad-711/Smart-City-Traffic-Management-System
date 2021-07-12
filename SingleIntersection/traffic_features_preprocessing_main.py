from os import path
import mpi4py
import numpy as np
import pandas as pd
import os
from mpi4py import MPI

from utils import  read_data_from_xml


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dataset_num = 2

    for index in range(rank * 3, (rank * 3) + 3):
        #reading data form files
        if index >= 100:
            break
        print('----- Readinding from data%i -----' % (index))
        path = os.path.join('TrafficData','traffic_data_%i' % (dataset_num))
        file_name = 'data' + str(index) + '.xml'
        traffic_data = []
        traffic_data.append(read_data_from_xml(os.path.join(path, file_name), index))
    
    



