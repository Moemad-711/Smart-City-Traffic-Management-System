from os import path
import mpi4py
import numpy as np
import pandas as pd
import os
from mpi4py import MPI

from PredictiveModel import PredictiveModel
from utils import  read_data_from_xml


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for index in range(80 + rank * 2, 80 + (rank * 2) + 2):
        #print('hello from rank %i: '%(rank), index)
        #reading data form files
        print('----- Readinding from data%i -----' % (index))
        file_name = 'data' + str(index) + '.xml'
        traffic_data = []
        traffic_data.append(read_data_from_xml(file_name,index))


